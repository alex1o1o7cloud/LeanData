import Mathlib

namespace NUMINAMATH_CALUDE_carnival_ticket_cost_l3337_333784

/-- Calculate the total cost of carnival tickets --/
theorem carnival_ticket_cost (kids_ticket_price : ℚ) (kids_ticket_quantity : ℕ)
  (adult_ticket_price : ℚ) (adult_ticket_quantity : ℕ)
  (kids_tickets_bought : ℕ) (adult_tickets_bought : ℕ) :
  kids_ticket_price * (kids_tickets_bought / kids_ticket_quantity : ℚ) +
  adult_ticket_price * (adult_tickets_bought / adult_ticket_quantity : ℚ) = 9 :=
by
  sorry

#check carnival_ticket_cost (1/4) 4 (2/3) 3 12 9

end NUMINAMATH_CALUDE_carnival_ticket_cost_l3337_333784


namespace NUMINAMATH_CALUDE_company_fund_problem_l3337_333783

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →  -- The fund initially contained $10 less than needed for $60 bonuses
  (initial_fund = 50 * n + 140) → -- Each employee received a $50 bonus, and $140 remained
  initial_fund = 890 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l3337_333783


namespace NUMINAMATH_CALUDE_correction_is_11y_l3337_333723

/-- The correction needed when y quarters are mistakenly counted as nickels
    and y pennies are mistakenly counted as dimes -/
def correction (y : ℕ) : ℤ :=
  let quarter_value : ℕ := 25
  let nickel_value : ℕ := 5
  let penny_value : ℕ := 1
  let dime_value : ℕ := 10
  let quarter_nickel_diff : ℕ := quarter_value - nickel_value
  let dime_penny_diff : ℕ := dime_value - penny_value
  (quarter_nickel_diff * y : ℤ) - (dime_penny_diff * y : ℤ)

theorem correction_is_11y (y : ℕ) : correction y = 11 * y :=
  sorry

end NUMINAMATH_CALUDE_correction_is_11y_l3337_333723


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l3337_333746

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of Quadrant I in the Cartesian plane -/
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The main theorem stating that the given linear function does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I (f : LinearFunction) 
  (h1 : f.m = -2)
  (h2 : f.b = -1) : 
  ¬ ∃ (x y : ℝ), y = f.m * x + f.b ∧ QuadrantI x y :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l3337_333746


namespace NUMINAMATH_CALUDE_triangle_properties_l3337_333793

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  c * Real.sin C / Real.sin A - c = b * Real.sin B / Real.sin A - a →
  b = 2 →
  (B = π / 3 ∧
   (a = 2 * Real.sqrt 6 / 3 →
    1/2 * a * b * Real.sin C = 1 + Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3337_333793


namespace NUMINAMATH_CALUDE_h_more_efficient_l3337_333796

/-- The daily harvest rate of a K combine in hectares -/
def k_rate : ℝ := sorry

/-- The daily harvest rate of an H combine in hectares -/
def h_rate : ℝ := sorry

/-- The total harvest of 4 K combines and 3 H combines in 5 days -/
def harvest1 : ℝ := 5 * (4 * k_rate + 3 * h_rate)

/-- The total harvest of 3 K combines and 5 H combines in 4 days -/
def harvest2 : ℝ := 4 * (3 * k_rate + 5 * h_rate)

/-- The theorem stating that H combines harvest more per day than K combines -/
theorem h_more_efficient : harvest1 = harvest2 → h_rate > k_rate := by
  sorry

end NUMINAMATH_CALUDE_h_more_efficient_l3337_333796


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l3337_333779

/-- The quadratic function f(x) = x^2 + 14x + 6 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 6

/-- Theorem: The minimum value of f(x) occurs when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) := by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l3337_333779


namespace NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l3337_333739

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_non_coincident : non_coincident m n)
  (h_plane_non_coincident : plane_non_coincident α β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l3337_333739


namespace NUMINAMATH_CALUDE_interest_difference_relation_l3337_333725

/-- Represents the compound interest scenario -/
structure CompoundInterest where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate

/-- Calculate the difference in compound interest between year 2 and year 1 -/
def interestDifference (ci : CompoundInterest) : ℝ :=
  ci.P * ci.r^2

/-- The theorem stating the relationship between the original and tripled interest rate scenarios -/
theorem interest_difference_relation (ci : CompoundInterest) :
  interestDifference { P := ci.P, r := 3 * ci.r } = 360 →
  interestDifference ci = 40 :=
by
  sorry

#check interest_difference_relation

end NUMINAMATH_CALUDE_interest_difference_relation_l3337_333725


namespace NUMINAMATH_CALUDE_ticket_sales_revenue_l3337_333750

/-- The total money made from ticket sales given the conditions -/
def total_money_made (advance_price same_day_price total_tickets advance_tickets : ℕ) : ℕ :=
  advance_price * advance_tickets + same_day_price * (total_tickets - advance_tickets)

/-- Theorem stating that the total money made is $1600 under the given conditions -/
theorem ticket_sales_revenue : 
  total_money_made 20 30 60 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_revenue_l3337_333750


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3337_333727

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3337_333727


namespace NUMINAMATH_CALUDE_valid_numbers_l3337_333792

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧  -- 4-digit number
  (n % 6 = 0) ∧ (n % 7 = 0) ∧ (n % 8 = 0) ∧  -- divisible by 6, 7, and 8
  (n % 4 ≠ 0) ∧ (n % 3 ≠ 0) ∧  -- not divisible by 4 or 3
  (n / 100 = 55) ∧  -- first two digits are 55
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 22) ∧  -- sum of digits is 22
  (∃ (a b : Nat), n = a * 1100 + b * 11)  -- two digits repeat twice

theorem valid_numbers : 
  ∀ n : Nat, is_valid_number n ↔ (n = 5566 ∨ n = 6655) := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l3337_333792


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3337_333721

/-- The area of a stripe wrapped around a cylindrical silo. -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * Real.pi * diameter = 480 * Real.pi := by
  sorry

#check stripe_area_on_cylindrical_silo

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3337_333721


namespace NUMINAMATH_CALUDE_chapter_page_difference_l3337_333760

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 37)
  (h2 : second_chapter_pages = 80) : 
  second_chapter_pages - first_chapter_pages = 43 := by
sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l3337_333760


namespace NUMINAMATH_CALUDE_article_sale_price_l3337_333770

/-- Proves that the selling price incurring a loss equal to the profit from selling at 832 is 448,
    given the conditions stated in the problem. -/
theorem article_sale_price (cp : ℝ) : 
  (832 - cp = cp - 448) →  -- Profit from selling at 832 equals loss when sold at unknown amount
  (768 - cp = 0.2 * cp) →  -- Sale price for 20% profit is 768
  (448 : ℝ) = 832 - 2 * cp := by
  sorry

#check article_sale_price

end NUMINAMATH_CALUDE_article_sale_price_l3337_333770


namespace NUMINAMATH_CALUDE_power_product_equality_l3337_333786

theorem power_product_equality : (0.125^8 * (-8)^7) = -0.125 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3337_333786


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l3337_333785

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1

-- State the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0) → a ∈ Set.Ici (-1) :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l3337_333785


namespace NUMINAMATH_CALUDE_weight_difference_l3337_333735

/-- Given the weights of Heather and Emily, prove the difference in their weights -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h1 : heather_weight = 87)
  (h2 : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3337_333735


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3337_333707

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f (-2) < 0 := sorry
  have h3 : f (-1) > 0 := sorry
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_root_exists_in_interval_l3337_333707


namespace NUMINAMATH_CALUDE_kannon_apples_difference_kannon_apples_difference_proof_l3337_333788

theorem kannon_apples_difference : ℕ → Prop :=
  fun x => 
    let apples_last_night : ℕ := 3
    let bananas_last_night : ℕ := 1
    let oranges_last_night : ℕ := 4
    let apples_today : ℕ := x
    let bananas_today : ℕ := 10 * bananas_last_night
    let oranges_today : ℕ := 2 * apples_today
    let total_fruits : ℕ := 39
    (apples_last_night + bananas_last_night + oranges_last_night + 
     apples_today + bananas_today + oranges_today = total_fruits) →
    (apples_today > apples_last_night) →
    (apples_today - apples_last_night = 4)

-- Proof
theorem kannon_apples_difference_proof : kannon_apples_difference 7 := by
  sorry

end NUMINAMATH_CALUDE_kannon_apples_difference_kannon_apples_difference_proof_l3337_333788


namespace NUMINAMATH_CALUDE_regular_polygon_sides_and_exterior_angle_l3337_333703

/-- 
Theorem: For a regular polygon with n sides, if the sum of its interior angles 
is greater than the sum of its exterior angles by 360°, then n = 6 and each 
exterior angle measures 60°.
-/
theorem regular_polygon_sides_and_exterior_angle (n : ℕ) : 
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (180 * (n - 2) = 360 + 360) →  -- Sum of interior angles equals 360° + sum of exterior angles
  (n = 6 ∧ 360 / n = 60) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_and_exterior_angle_l3337_333703


namespace NUMINAMATH_CALUDE_existence_of_midpoint_with_odd_double_coordinates_l3337_333705

/-- A point in the xy-plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A sequence of 1993 distinct points with the required properties -/
def PointSequence : Type :=
  { ps : Fin 1993 → IntPoint //
    (∀ i j, i ≠ j → ps i ≠ ps j) ∧  -- points are distinct
    (∀ i : Fin 1992, ∀ p : IntPoint,
      p ≠ ps i ∧ p ≠ ps (i + 1) →
      ¬∃ (t : ℚ), 0 < t ∧ t < 1 ∧
        p.x = (1 - t) * (ps i).x + t * (ps (i + 1)).x ∧
        p.y = (1 - t) * (ps i).y + t * (ps (i + 1)).y) }

theorem existence_of_midpoint_with_odd_double_coordinates (ps : PointSequence) :
    ∃ i : Fin 1992, ∃ qx qy : ℚ,
      (2 * qx).num % 2 = 1 ∧
      (2 * qy).num % 2 = 1 ∧
      qx = ((ps.val i).x + (ps.val (i + 1)).x) / 2 ∧
      qy = ((ps.val i).y + (ps.val (i + 1)).y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_midpoint_with_odd_double_coordinates_l3337_333705


namespace NUMINAMATH_CALUDE_extremum_and_inequality_l3337_333720

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) * Real.log (x + a)

theorem extremum_and_inequality (h : ∀ a > 0, ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a x ≤ f a 0) :
  (∃ a > 0, (deriv (f a)) 0 = 0) ∧
  (∀ x ≥ 0, f 1 x ≥ x^2) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_inequality_l3337_333720


namespace NUMINAMATH_CALUDE_quadrilateral_properties_exist_l3337_333771

noncomputable def quadrilateral_properties (a b c d t : ℝ) : Prop :=
  ∃ (α β γ δ ε : ℝ) (e f : ℝ),
    α + β + γ + δ = 2 * Real.pi ∧
    a * d * Real.sin α + b * c * Real.sin γ = 2 * t ∧
    a * b * Real.sin β + c * d * Real.sin δ = 2 * t ∧
    e^2 = a^2 + b^2 - 2*a*b * Real.cos β ∧
    e^2 = c^2 + d^2 - 2*c*d * Real.cos δ ∧
    f^2 = a^2 + d^2 - 2*a*d * Real.cos α ∧
    f^2 = b^2 + c^2 - 2*b*c * Real.cos γ ∧
    t = (1/2) * e * f * Real.sin ε

theorem quadrilateral_properties_exist (a b c d t : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (ht : t > 0) :
  quadrilateral_properties a b c d t :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_exist_l3337_333771


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3337_333718

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3337_333718


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3337_333795

/-- The axis of symmetry of a parabola y = a(x+1)(x-3) where a ≠ 0 -/
def axisOfSymmetry (a : ℝ) (h : a ≠ 0) : ℝ := 1

/-- Theorem stating that the axis of symmetry of the parabola y = a(x+1)(x-3) where a ≠ 0 is x = 1 -/
theorem parabola_axis_of_symmetry (a : ℝ) (h : a ≠ 0) :
  axisOfSymmetry a h = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3337_333795


namespace NUMINAMATH_CALUDE_time_rosa_sees_leo_l3337_333710

/-- Calculates the time Rosa can see Leo given their speeds and distances -/
theorem time_rosa_sees_leo (rosa_speed leo_speed initial_distance final_distance : ℚ) :
  rosa_speed = 15 →
  leo_speed = 5 →
  initial_distance = 3/4 →
  final_distance = 3/4 →
  (initial_distance + final_distance) / (rosa_speed - leo_speed) * 60 = 9 := by
  sorry

#check time_rosa_sees_leo

end NUMINAMATH_CALUDE_time_rosa_sees_leo_l3337_333710


namespace NUMINAMATH_CALUDE_playlist_song_length_l3337_333777

theorem playlist_song_length 
  (n_short_songs : ℕ) 
  (short_song_length : ℕ) 
  (n_long_songs : ℕ) 
  (total_duration : ℕ) 
  (additional_time_needed : ℕ) 
  (h1 : n_short_songs = 10)
  (h2 : short_song_length = 3)
  (h3 : n_long_songs = 15)
  (h4 : total_duration = 100)
  (h5 : additional_time_needed = 40) :
  ∃ (long_song_length : ℚ),
    long_song_length = 14/3 ∧ 
    n_short_songs * short_song_length + n_long_songs * long_song_length = total_duration := by
  sorry

end NUMINAMATH_CALUDE_playlist_song_length_l3337_333777


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l3337_333762

def calculate_remaining_cakes (initial_cakes : ℕ) (sold_cakes : ℕ) (additional_cakes : ℕ) : ℕ :=
  initial_cakes - sold_cakes + additional_cakes

theorem baker_cakes_theorem (initial_cakes sold_cakes additional_cakes : ℕ) 
  (h1 : initial_cakes ≥ sold_cakes) :
  calculate_remaining_cakes initial_cakes sold_cakes additional_cakes = 
  initial_cakes - sold_cakes + additional_cakes :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l3337_333762


namespace NUMINAMATH_CALUDE_disinfectant_transport_theorem_l3337_333744

/-- Represents the number of bottles a box can hold -/
structure BoxCapacity where
  large : Nat
  small : Nat

/-- Represents the cost of a box in yuan -/
structure BoxCost where
  large : Nat
  small : Nat

/-- Represents the carrying capacity of a vehicle -/
structure VehicleCapacity where
  large : Nat
  small : Nat

/-- Represents the number of boxes purchased -/
structure Boxes where
  large : Nat
  small : Nat

/-- Represents the number of vehicles of each type -/
structure Vehicles where
  typeA : Nat
  typeB : Nat

def total_bottles : Nat := 3250
def total_cost : Nat := 1700
def total_vehicles : Nat := 10

def box_capacity : BoxCapacity := { large := 10, small := 5 }
def box_cost : BoxCost := { large := 5, small := 3 }
def vehicle_capacity_A : VehicleCapacity := { large := 30, small := 10 }
def vehicle_capacity_B : VehicleCapacity := { large := 20, small := 40 }

def is_valid_box_purchase (boxes : Boxes) : Prop :=
  boxes.large * box_capacity.large + boxes.small * box_capacity.small = total_bottles ∧
  boxes.large * box_cost.large + boxes.small * box_cost.small = total_cost

def is_valid_vehicle_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  vehicles.typeA + vehicles.typeB = total_vehicles ∧
  vehicles.typeA * vehicle_capacity_A.large + vehicles.typeB * vehicle_capacity_B.large ≥ boxes.large ∧
  vehicles.typeA * vehicle_capacity_A.small + vehicles.typeB * vehicle_capacity_B.small ≥ boxes.small

def is_optimal_arrangement (vehicles : Vehicles) (boxes : Boxes) : Prop :=
  is_valid_vehicle_arrangement vehicles boxes ∧
  ∀ (other : Vehicles), is_valid_vehicle_arrangement other boxes → vehicles.typeA ≥ other.typeA

theorem disinfectant_transport_theorem : 
  ∃ (boxes : Boxes) (vehicles : Vehicles),
    is_valid_box_purchase boxes ∧
    boxes.large = 250 ∧
    boxes.small = 150 ∧
    is_optimal_arrangement vehicles boxes ∧
    vehicles.typeA = 8 ∧
    vehicles.typeB = 2 := by sorry

end NUMINAMATH_CALUDE_disinfectant_transport_theorem_l3337_333744


namespace NUMINAMATH_CALUDE_fraction_power_product_equals_three_halves_l3337_333732

theorem fraction_power_product_equals_three_halves :
  (3 / 2 : ℝ) ^ 2023 * (2 / 3 : ℝ) ^ 2022 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_equals_three_halves_l3337_333732


namespace NUMINAMATH_CALUDE_count_solutions_l3337_333741

def positive_integer_solutions : Nat :=
  let n := 25
  let k := 5
  let min_values := [2, 3, 1, 2, 4]
  let remaining := n - (min_values.sum)
  Nat.choose (remaining + k - 1) (k - 1)

theorem count_solutions :
  positive_integer_solutions = 1190 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_l3337_333741


namespace NUMINAMATH_CALUDE_dice_probability_l3337_333736

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Predicate to check if a number is a multiple of 5 -/
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- The set of all possible outcomes when rolling num_dice dice -/
def all_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where at least one die shows an even number -/
def even_product_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where the sum of dice is a multiple of 5 -/
def sum_multiple_of_5_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The number of favorable outcomes (sum is multiple of 5 given product is even) -/
def a : ℕ := sorry

/-- The probability of the sum being a multiple of 5 given the product is even -/
theorem dice_probability : 
  (Finset.card (even_product_outcomes ∩ sum_multiple_of_5_outcomes) : ℚ) / 
  (Finset.card even_product_outcomes : ℚ) = 
  (a : ℚ) / ((num_sides ^ num_dice - (num_sides / 2) ^ num_dice) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l3337_333736


namespace NUMINAMATH_CALUDE_interest_rate_increase_l3337_333757

theorem interest_rate_increase (initial_rate : ℝ) (increase_percentage : ℝ) (final_rate : ℝ) : 
  initial_rate = 8.256880733944953 →
  increase_percentage = 10 →
  final_rate = initial_rate * (1 + increase_percentage / 100) →
  final_rate = 9.082568807339448 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_increase_l3337_333757


namespace NUMINAMATH_CALUDE_arithmetic_sequence_4_to_256_l3337_333713

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The last term of an arithmetic sequence -/
def arithmetic_sequence_last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_4_to_256 :
  arithmetic_sequence_length 4 4 256 = 64 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_4_to_256_l3337_333713


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3337_333763

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3337_333763


namespace NUMINAMATH_CALUDE_regular_ngon_triangle_property_l3337_333719

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Triangle type: acute, right, or obtuse -/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- Determine the type of a triangle given its vertices -/
def triangleType (A B C : ℝ × ℝ) : TriangleType :=
  sorry

/-- The main theorem -/
theorem regular_ngon_triangle_property (n : ℕ) (hn : n > 0) :
  ∀ (P : RegularNGon n) (σ : Fin n → Fin n),
  Function.Bijective σ →
  ∃ (i j k : Fin n),
    triangleType (P.vertices i) (P.vertices j) (P.vertices k) =
    triangleType (P.vertices (σ i)) (P.vertices (σ j)) (P.vertices (σ k)) :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_triangle_property_l3337_333719


namespace NUMINAMATH_CALUDE_one_third_of_1206_percent_of_200_l3337_333794

theorem one_third_of_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_1206_percent_of_200_l3337_333794


namespace NUMINAMATH_CALUDE_inequality_proof_l3337_333774

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3337_333774


namespace NUMINAMATH_CALUDE_tank_insulation_problem_l3337_333767

/-- Proves that for a rectangular tank with given dimensions and insulation cost, 
    the third dimension is 2 feet. -/
theorem tank_insulation_problem (x : ℝ) : 
  x > 0 → 
  (2 * 3 * 5 + 2 * 3 * x + 2 * 5 * x) * 20 = 1240 → 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_tank_insulation_problem_l3337_333767


namespace NUMINAMATH_CALUDE_no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l3337_333711

theorem no_solution_for_k_2_and_3 :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) :=
sorry

theorem solution_exists_for_k_ge_4 :
  ∀ (k : ℕ), k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l3337_333711


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3337_333791

/-- Given a quadratic expression 3x^2 + 9x - 24, when written in the form a(x - h)^2 + k, h = -1.5 -/
theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x - 24 = a*(x - (-1.5))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3337_333791


namespace NUMINAMATH_CALUDE_gcd_1260_924_l3337_333734

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1260_924_l3337_333734


namespace NUMINAMATH_CALUDE_show_revenue_calculation_l3337_333764

def first_showing_attendance : ℕ := 200
def second_showing_multiplier : ℕ := 3
def ticket_price : ℕ := 25

theorem show_revenue_calculation :
  let second_showing_attendance := first_showing_attendance * second_showing_multiplier
  let total_attendance := first_showing_attendance + second_showing_attendance
  let total_revenue := total_attendance * ticket_price
  total_revenue = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_calculation_l3337_333764


namespace NUMINAMATH_CALUDE_num_cows_bought_l3337_333712

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals Zara bought -/
def total_animals : ℕ := num_groups * animals_per_group

theorem num_cows_bought : 
  total_animals - (num_sheep + num_goats) = 24 :=
by sorry

end NUMINAMATH_CALUDE_num_cows_bought_l3337_333712


namespace NUMINAMATH_CALUDE_range_of_m_l3337_333755

-- Define the propositions p and q
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (sufficient_not_necessary (p) (q m)) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3337_333755


namespace NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l3337_333756

theorem negation_of_square_positive_equals_zero :
  (¬ ∀ m : ℝ, m > 0 → m^2 = 0) ↔ (∀ m : ℝ, m ≤ 0 → m^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l3337_333756


namespace NUMINAMATH_CALUDE_composition_equality_l3337_333790

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + a
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 + 1

-- State the theorem
theorem composition_equality (a : ℝ) :
  ∃ b : ℝ, ∀ x : ℝ, f a (g a x) = a^2*x^4 + 5*a*x^2 + b → b = 6 + a :=
by sorry

end NUMINAMATH_CALUDE_composition_equality_l3337_333790


namespace NUMINAMATH_CALUDE_total_defective_rate_l3337_333758

/-- Given two workers x and y who check products, with known defective rates and
    the fraction of products checked by worker y, prove the total defective rate. -/
theorem total_defective_rate 
  (defective_rate_x : ℝ) 
  (defective_rate_y : ℝ) 
  (fraction_checked_by_y : ℝ) 
  (h1 : defective_rate_x = 0.005) 
  (h2 : defective_rate_y = 0.008) 
  (h3 : fraction_checked_by_y = 0.5) 
  (h4 : fraction_checked_by_y ≥ 0 ∧ fraction_checked_by_y ≤ 1) : 
  defective_rate_x * (1 - fraction_checked_by_y) + defective_rate_y * fraction_checked_by_y = 0.0065 := by
  sorry

#check total_defective_rate

end NUMINAMATH_CALUDE_total_defective_rate_l3337_333758


namespace NUMINAMATH_CALUDE_min_value_expression_l3337_333738

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a + b) / c + (a + c) / b + (b + c) / a ≥ 8 ∧
  (2 * (a + b) / c + (a + c) / b + (b + c) / a = 8 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3337_333738


namespace NUMINAMATH_CALUDE_interest_calculation_l3337_333731

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem interest_calculation (principal : ℚ) (rate : ℚ) (time : ℚ) 
  (h1 : principal = 3000)
  (h2 : rate = 5)
  (h3 : time = 5)
  (h4 : simple_interest principal rate time = principal - 2250) :
  simple_interest principal rate time = 750 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3337_333731


namespace NUMINAMATH_CALUDE_triangle_properties_l3337_333704

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A ∧
  t.b + t.c = Real.sqrt 10 ∧
  t.a = 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3337_333704


namespace NUMINAMATH_CALUDE_coin_arrangement_concyclic_l3337_333737

-- Define the circles (coins)
variable (O₁ O₂ O₃ O₄ : ℝ × ℝ)  -- Centers of the circles
variable (r₁ r₂ r₃ r₄ : ℝ)      -- Radii of the circles

-- Define the points of intersection
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the property of being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem coin_arrangement_concyclic :
  concyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_coin_arrangement_concyclic_l3337_333737


namespace NUMINAMATH_CALUDE_min_square_size_and_unused_area_l3337_333761

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The shapes contained within the larger square -/
def contained_shapes : List Rectangle := [
  { width := 2, height := 2 },  -- 2x2 square
  { width := 1, height := 3 },  -- 1x3 rectangle
  { width := 2, height := 1 }   -- 2x1 rectangle
]

/-- Theorem: The minimum side length of the containing square is 5,
    and the minimum unused area is 16 -/
theorem min_square_size_and_unused_area :
  let min_side := 5
  let total_area := min_side * min_side
  let shapes_area := (contained_shapes.map Rectangle.area).sum
  let unused_area := total_area - shapes_area
  (∀ side : ℕ, side ≥ min_side → 
    side * side - shapes_area ≥ unused_area) ∧
  unused_area = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_square_size_and_unused_area_l3337_333761


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3337_333716

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3337_333716


namespace NUMINAMATH_CALUDE_probability_three_colors_l3337_333722

/-- The probability of picking at least one ball of each color when selecting 3 balls from a jar
    containing 8 black, 5 white, and 3 red balls is 3/14. -/
theorem probability_three_colors (black white red : ℕ) (total : ℕ) (h1 : black = 8) (h2 : white = 5) (h3 : red = 3) 
    (h4 : total = black + white + red) : 
  (black * white * red : ℚ) / (total * (total - 1) * (total - 2) / 6) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_colors_l3337_333722


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l3337_333789

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 5, 1]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 233 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_age_conversion_l3337_333789


namespace NUMINAMATH_CALUDE_minutes_to_seconds_conversion_seconds_to_minutes_conversion_l3337_333701

-- Define the conversion factor
def seconds_per_minute : ℝ := 60

-- Define the number of minutes
def minutes : ℝ := 8.5

-- Theorem to prove
theorem minutes_to_seconds_conversion :
  minutes * seconds_per_minute = 510 := by
  sorry

-- Verification theorem
theorem seconds_to_minutes_conversion :
  510 / seconds_per_minute = minutes := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_conversion_seconds_to_minutes_conversion_l3337_333701


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3337_333733

theorem rectangle_perimeter (a b : ℝ) :
  let area := 3 * a^2 - 3 * a * b + 6 * a
  let side1 := 3 * a
  let side2 := area / side1
  side1 > 0 → side2 > 0 →
  2 * (side1 + side2) = 8 * a - 2 * b + 4 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3337_333733


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3337_333769

def set_A (a : ℝ) : Set ℝ := {-2, 3*a-1, a^2-3}
def set_B (a : ℝ) : Set ℝ := {a-2, a-1, a+1}

theorem intersection_implies_a_value (a : ℝ) :
  set_A a ∩ set_B a = {-2} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3337_333769


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3337_333709

theorem nested_square_root_value : ∃ y : ℝ, y > 0 ∧ y = Real.sqrt (3 - y) ∧ y = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3337_333709


namespace NUMINAMATH_CALUDE_equation_system_solution_l3337_333780

def equation_system (x y z : ℝ) : Prop :=
  x^2 + y + z = 1 ∧ x + y^2 + z = 1 ∧ x + y + z^2 = 1

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 0, 0), (0, 1, 0), (0, 0, 1), 
   (-1 - Real.sqrt 2, -1 - Real.sqrt 2, -1 - Real.sqrt 2),
   (-1 + Real.sqrt 2, -1 + Real.sqrt 2, -1 + Real.sqrt 2)}

theorem equation_system_solution :
  ∀ x y z : ℝ, equation_system x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3337_333780


namespace NUMINAMATH_CALUDE_chad_cracker_boxes_l3337_333724

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of nights the crackers will last -/
def nights_lasting : ℕ := 56

/-- Calculates the number of boxes of crackers Chad has -/
def boxes_of_crackers : ℕ :=
  (crackers_per_sandwich * sandwiches_per_night * nights_lasting) /
  (sleeves_per_box * crackers_per_sleeve)

theorem chad_cracker_boxes :
  boxes_of_crackers = 5 := by
  sorry

end NUMINAMATH_CALUDE_chad_cracker_boxes_l3337_333724


namespace NUMINAMATH_CALUDE_system_solution_unique_l3337_333765

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x + 3 * y = 7 ∧ 4 * x - 3 * y = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3337_333765


namespace NUMINAMATH_CALUDE_empty_set_implies_m_zero_l3337_333743

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_m_zero_l3337_333743


namespace NUMINAMATH_CALUDE_first_car_speed_l3337_333773

/-- Represents the scenario of two cars traveling between points A and B -/
structure CarScenario where
  distance_AB : ℝ
  delay : ℝ
  speed_second_car : ℝ
  speed_first_car : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def satisfies_conditions (s : CarScenario) : Prop :=
  s.distance_AB = 40 ∧
  s.delay = 1/3 ∧
  s.speed_second_car = 45 ∧
  ∃ (meeting_point : ℝ),
    0 < meeting_point ∧ meeting_point < s.distance_AB ∧
    (meeting_point / s.speed_second_car + s.delay = meeting_point / s.speed_first_car) ∧
    (s.distance_AB / s.speed_first_car = 
      meeting_point / s.speed_second_car + s.delay + meeting_point / s.speed_second_car + meeting_point / (2 * s.speed_second_car))

/-- The main theorem stating that if a scenario satisfies all conditions, 
    then the speed of the first car must be 30 km/h -/
theorem first_car_speed (s : CarScenario) :
  satisfies_conditions s → s.speed_first_car = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l3337_333773


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l3337_333778

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- State the theorem
theorem angle_sum_in_triangle (t : Triangle) 
  (h1 : t.A = 65)
  (h2 : t.B = 40) : 
  t.C = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l3337_333778


namespace NUMINAMATH_CALUDE_homework_time_distribution_l3337_333726

theorem homework_time_distribution (total_time : ℕ) (math_percent : ℚ) (science_percent : ℚ) 
  (h1 : total_time = 150)
  (h2 : math_percent = 30 / 100)
  (h3 : science_percent = 40 / 100) :
  total_time - (math_percent * total_time + science_percent * total_time) = 45 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_distribution_l3337_333726


namespace NUMINAMATH_CALUDE_inverse_function_point_l3337_333748

open Real

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  tan (π / 3) - f 2 = Real.sqrt 3 - 1 / 3 →
  Function.invFun f (1 / 3) - π / 2 = 2 - π / 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l3337_333748


namespace NUMINAMATH_CALUDE_solution_comparison_l3337_333714

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
  (h_sol : -q / p > -q' / p') : q / p < q' / p' := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l3337_333714


namespace NUMINAMATH_CALUDE_triangle_theorem_l3337_333700

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with geometric sequence sides -/
theorem triangle_theorem (t : Triangle) 
  (geom_seq : t.b^2 = t.a * t.c)
  (cos_B : Real.cos t.B = 3/5)
  (area : 1/2 * t.a * t.c * Real.sin t.B = 2) :
  (t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 21) ∧
  ((Real.sqrt 5 - 1)/2 < (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
                         (Real.sin t.B + Real.cos t.B * Real.tan t.C) ∧
   (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
   (Real.sin t.B + Real.cos t.B * Real.tan t.C) < (Real.sqrt 5 + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3337_333700


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3337_333781

-- Define a triangle with angles in 2:3:4 ratio
def triangle_with_ratio (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b = (3/2) * a ∧ c = 2 * a ∧
  a + b + c = 180

-- Theorem statement
theorem smallest_angle_measure (a b c : ℝ) 
  (h : triangle_with_ratio a b c) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3337_333781


namespace NUMINAMATH_CALUDE_total_routes_bristol_to_carlisle_l3337_333745

theorem total_routes_bristol_to_carlisle :
  let bristol_to_birmingham : ℕ := 8
  let birmingham_to_manchester : ℕ := 5
  let manchester_to_sheffield : ℕ := 4
  let sheffield_to_newcastle : ℕ := 3
  let newcastle_to_carlisle : ℕ := 2
  bristol_to_birmingham * birmingham_to_manchester * manchester_to_sheffield * sheffield_to_newcastle * newcastle_to_carlisle = 960 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_bristol_to_carlisle_l3337_333745


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l3337_333715

theorem sin_five_pi_sixths_minus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l3337_333715


namespace NUMINAMATH_CALUDE_tire_price_problem_l3337_333797

theorem tire_price_problem (regular_price : ℝ) : 
  (3 * regular_price + 10 = 310) → regular_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l3337_333797


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3337_333747

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter (Nat.Prime) (Finset.range 31)).card = 10 ∧
  ∀ p : ℕ, Nat.Prime p → p ∣ factorial 30 ↔ p ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3337_333747


namespace NUMINAMATH_CALUDE_alice_bob_meet_l3337_333742

/-- The number of points on the circular track -/
def n : ℕ := 15

/-- Alice's movement in clockwise direction per turn -/
def a : ℕ := 7

/-- Bob's movement in counterclockwise direction per turn -/
def b : ℕ := 10

/-- The function that calculates the position after k turns -/
def position (movement : ℕ) (k : ℕ) : ℕ :=
  (movement * k) % n

/-- The theorem stating that Alice and Bob meet after 8 turns -/
theorem alice_bob_meet :
  (∀ k : ℕ, k < 8 → position a k ≠ position (n - b) k) ∧
  position a 8 = position (n - b) 8 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l3337_333742


namespace NUMINAMATH_CALUDE_gum_pack_size_l3337_333730

theorem gum_pack_size (initial_peach : ℕ) (initial_mint : ℕ) (y : ℚ) 
  (h1 : initial_peach = 40)
  (h2 : initial_mint = 50)
  (h3 : y > 0) :
  (initial_peach - 2 * y) / initial_mint = initial_peach / (initial_mint + 3 * y) → 
  y = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l3337_333730


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l3337_333754

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → bought_cards = 224 → remaining_cards = initial_cards - bought_cards → 
  remaining_cards = 452 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l3337_333754


namespace NUMINAMATH_CALUDE_medical_team_selection_count_l3337_333766

theorem medical_team_selection_count : ∀ (m f k l : ℕ), 
  m = 6 → f = 5 → k = 2 → l = 1 →
  (m.choose k) * (f.choose l) = 75 :=
by sorry

end NUMINAMATH_CALUDE_medical_team_selection_count_l3337_333766


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l3337_333782

theorem sum_of_five_consecutive_even_integers (n : ℤ) :
  (2*n) + (2*n + 2) + (2*n + 4) + (2*n + 6) + (2*n + 8) = 10*n + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l3337_333782


namespace NUMINAMATH_CALUDE_certain_number_proof_l3337_333740

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ x * 1.6 = 340.8 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3337_333740


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_side_ratio_l3337_333706

theorem triangle_with_arithmetic_angles_and_side_ratio (α β γ : Real) (a b c : Real) :
  -- Angles form an arithmetic progression
  β - α = γ - β →
  -- Sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- Smallest side is half of the largest side
  a = c / 2 →
  -- Side lengths satisfy the sine law
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  -- Angles are positive
  0 < α ∧ 0 < β ∧ 0 < γ →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Prove that the angles are 30°, 60°, and 90°
  (α = 30 ∧ β = 60 ∧ γ = 90) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_side_ratio_l3337_333706


namespace NUMINAMATH_CALUDE_february_2020_average_rainfall_l3337_333702

/-- Calculate the average rainfall per hour in February 2020 --/
theorem february_2020_average_rainfall
  (total_rainfall : ℝ)
  (february_days : ℕ)
  (hours_per_day : ℕ)
  (h1 : total_rainfall = 290)
  (h2 : february_days = 29)
  (h3 : hours_per_day = 24) :
  total_rainfall / (february_days * hours_per_day : ℝ) = 290 / 696 :=
by sorry

end NUMINAMATH_CALUDE_february_2020_average_rainfall_l3337_333702


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_square_l3337_333708

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating that 19 is the smallest two-digit prime number
    whose reverse is a perfect square -/
theorem smallest_two_digit_prime_reverse_square :
  (∀ n : ℕ, 10 ≤ n ∧ n < 19 ∧ is_prime n → ¬(is_square (reverse_digits n))) ∧
  (19 ≤ 99 ∧ is_prime 19 ∧ is_square (reverse_digits 19)) :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_square_l3337_333708


namespace NUMINAMATH_CALUDE_spell_casting_contest_orders_l3337_333729

/-- The number of different possible orders for a given number of competitors -/
def possibleOrders (n : ℕ) : ℕ := Nat.factorial n

/-- The number of competitors in the contest -/
def numberOfCompetitors : ℕ := 4

theorem spell_casting_contest_orders :
  possibleOrders numberOfCompetitors = 24 := by
  sorry

end NUMINAMATH_CALUDE_spell_casting_contest_orders_l3337_333729


namespace NUMINAMATH_CALUDE_inequality_count_l3337_333775

theorem inequality_count (x y a b : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_x_lt_a : x < a) (h_y_lt_b : y < b) : 
  (((x + y < a + b) ∧ (x * y < a * b) ∧ (x / y < a / b)) ∧ 
   ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x < a → y < b → x - y < a - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_count_l3337_333775


namespace NUMINAMATH_CALUDE_aperture_radius_ratio_l3337_333776

theorem aperture_radius_ratio (r : ℝ) (h : r > 0) : 
  ∃ (r_new : ℝ), (π * r_new^2 = 2 * π * r^2) ∧ (r_new / r = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_aperture_radius_ratio_l3337_333776


namespace NUMINAMATH_CALUDE_P_root_nature_l3337_333772

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^5 - 4*x^4 - 6*x^3 - x + 8

-- Theorem stating that P(x) has no negative roots and at least one positive root
theorem P_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by
  sorry


end NUMINAMATH_CALUDE_P_root_nature_l3337_333772


namespace NUMINAMATH_CALUDE_coffee_blend_type_A_quantity_l3337_333759

/-- Represents the cost and quantity of coffee types in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem coffee_blend_type_A_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end NUMINAMATH_CALUDE_coffee_blend_type_A_quantity_l3337_333759


namespace NUMINAMATH_CALUDE_min_female_participants_l3337_333799

theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = (male_students + female_students) * 60 / 100) :
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by
  sorry

end NUMINAMATH_CALUDE_min_female_participants_l3337_333799


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l3337_333753

theorem bus_stoppage_time (s1 s2 s3 v1 v2 v3 : ℝ) 
  (h1 : s1 = 54) (h2 : s2 = 60) (h3 : s3 = 72)
  (h4 : v1 = 36) (h5 : v2 = 40) (h6 : v3 = 48) :
  (1 - v1 / s1) + (1 - v2 / s2) + (1 - v3 / s3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l3337_333753


namespace NUMINAMATH_CALUDE_base_k_is_seven_l3337_333787

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := 
  (n / 100) * k^2 + ((n / 10) % 10) * k + (n % 10)

/-- The theorem stating that 7 is the base k where (524)₈ = (664)ₖ -/
theorem base_k_is_seven : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 524 = baseKToBase10 664 k → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_k_is_seven_l3337_333787


namespace NUMINAMATH_CALUDE_extrema_of_sum_l3337_333728

theorem extrema_of_sum (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  let P := x + y
  (9 + 3 * Real.sqrt 21) / 2 ≤ P ∧ P ≤ 9 + 3 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_extrema_of_sum_l3337_333728


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l3337_333768

open Real

/-- The area enclosed by the locus of highest points of projectile trajectories. -/
theorem projectile_trajectory_area (v g : ℝ) (h : ℝ := v^2 / (8 * g)) : 
  ∃ (area : ℝ), area = (3 * π / 32) * (v^4 / g^2) :=
by sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l3337_333768


namespace NUMINAMATH_CALUDE_harry_work_hours_l3337_333798

/-- Given the payment conditions for Harry and James, prove that if James worked 41 hours
    and they were paid the same amount, then Harry worked 39 hours. -/
theorem harry_work_hours (x : ℝ) (h : ℝ) :
  let harry_pay := 24 * x + (h - 24) * 1.5 * x
  let james_pay := 24 * x + (41 - 24) * 2 * x
  harry_pay = james_pay →
  h = 39 := by
  sorry

end NUMINAMATH_CALUDE_harry_work_hours_l3337_333798


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3337_333752

theorem imaginary_part_of_z (m : ℝ) : 
  let z : ℂ := 1 - m * Complex.I
  (z ^ 2 = -2 * Complex.I) → (z.im = -1) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3337_333752


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l3337_333717

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l3337_333717


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l3337_333751

-- Define the probability of having a boy or a girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l3337_333751


namespace NUMINAMATH_CALUDE_investment_income_l3337_333749

theorem investment_income
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (h1 : total_investment = 8000)
  (h2 : first_investment = 3000)
  (h3 : first_rate = 0.085)
  (h4 : second_rate = 0.064) :
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 575 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_l3337_333749
