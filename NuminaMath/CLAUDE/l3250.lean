import Mathlib

namespace NUMINAMATH_CALUDE_equation_system_solution_equality_l3250_325092

theorem equation_system_solution_equality (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = r →
  5 * x + 3 * y = s →
  r - s = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_equality_l3250_325092


namespace NUMINAMATH_CALUDE_workshop_attendance_l3250_325070

/-- Represents the number of scientists at a workshop with various prize distributions -/
structure WorkshopAttendance where
  total : ℕ
  wolfPrize : ℕ
  nobelPrize : ℕ
  wolfAndNobel : ℕ

/-- Theorem stating the total number of scientists at the workshop -/
theorem workshop_attendance (w : WorkshopAttendance) 
  (h1 : w.wolfPrize = 31)
  (h2 : w.wolfAndNobel = 12)
  (h3 : w.nobelPrize = 23)
  (h4 : w.nobelPrize - w.wolfAndNobel = (w.total - w.wolfPrize - (w.nobelPrize - w.wolfAndNobel)) + 3) :
  w.total = 39 := by
  sorry


end NUMINAMATH_CALUDE_workshop_attendance_l3250_325070


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3250_325031

theorem unique_solution_cube_equation (x : ℝ) (h : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 ↔ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3250_325031


namespace NUMINAMATH_CALUDE_total_dolls_l3250_325067

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that the total number of dolls they have together is 48. -/
theorem total_dolls (hannah_multiplier sister_dolls : ℕ) 
  (h1 : hannah_multiplier = 5)
  (h2 : sister_dolls = 8) :
  hannah_multiplier * sister_dolls + sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l3250_325067


namespace NUMINAMATH_CALUDE_complex_power_difference_l3250_325019

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : 
  (1 + i)^18 - (1 - i)^18 = 1024 * i :=
sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3250_325019


namespace NUMINAMATH_CALUDE_translation_left_proof_l3250_325011

def translate_left (x y : ℝ) (d : ℝ) : ℝ × ℝ :=
  (x - d, y)

theorem translation_left_proof :
  let A : ℝ × ℝ := (1, 2)
  let A₁ : ℝ × ℝ := translate_left A.1 A.2 1
  A₁ = (0, 2) := by sorry

end NUMINAMATH_CALUDE_translation_left_proof_l3250_325011


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l3250_325054

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem monotonic_decreasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l3250_325054


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l3250_325060

theorem cats_remaining_after_sale
  (initial_siamese : ℕ)
  (initial_persian : ℕ)
  (initial_house : ℕ)
  (sold_siamese : ℕ)
  (sold_persian : ℕ)
  (sold_house : ℕ)
  (h1 : initial_siamese = 20)
  (h2 : initial_persian = 12)
  (h3 : initial_house = 8)
  (h4 : sold_siamese = 8)
  (h5 : sold_persian = 5)
  (h6 : sold_house = 3) :
  initial_siamese + initial_persian + initial_house -
  (sold_siamese + sold_persian + sold_house) = 24 :=
by sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l3250_325060


namespace NUMINAMATH_CALUDE_part1_part2_l3250_325027

-- Part 1
theorem part1 (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x + 1) = x + 2 * Real.sqrt x) →
  (∀ x ≥ 1, f x = x^2 - 2*x) :=
sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) →
  (∀ x, f x = 2 * x + 7) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3250_325027


namespace NUMINAMATH_CALUDE_attendance_problem_l3250_325009

/-- Proves that the number of people who didn't show up is 12 --/
theorem attendance_problem (total_invited : ℕ) (tables_used : ℕ) (table_capacity : ℕ) : 
  total_invited - (tables_used * table_capacity) = 12 :=
by
  sorry

#check attendance_problem 18 2 3

end NUMINAMATH_CALUDE_attendance_problem_l3250_325009


namespace NUMINAMATH_CALUDE_inequality_proof_l3250_325040

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 * (a ^ (1/2)) + 3 * (b ^ (1/3)) ≥ 5 * ((a * b) ^ (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3250_325040


namespace NUMINAMATH_CALUDE_ceiling_light_ratio_l3250_325063

/-- Represents the number of bulbs required for each type of ceiling light -/
structure BulbRequirement where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the counts of different types of ceiling lights -/
structure CeilingLightCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of bulbs required -/
def totalBulbs (req : BulbRequirement) (counts : CeilingLightCounts) : Nat :=
  req.small * counts.small + req.medium * counts.medium + req.large * counts.large

/-- Theorem statement for the ceiling light problem -/
theorem ceiling_light_ratio 
  (req : BulbRequirement)
  (counts : CeilingLightCounts)
  (h1 : req.small = 1 ∧ req.medium = 2 ∧ req.large = 3)
  (h2 : counts.medium = 12)
  (h3 : counts.small = counts.medium + 10)
  (h4 : totalBulbs req counts = 118) :
  counts.large = 2 * counts.medium := by
  sorry

#check ceiling_light_ratio

end NUMINAMATH_CALUDE_ceiling_light_ratio_l3250_325063


namespace NUMINAMATH_CALUDE_wuhan_spring_temp_difference_l3250_325018

/-- The average daily high temperature in spring in the Wuhan area -/
def average_high : ℝ := 15

/-- The lowest temperature in spring in the Wuhan area -/
def lowest_temp : ℝ := 7

/-- The difference between the average daily high temperature and the lowest temperature -/
def temp_difference : ℝ := average_high - lowest_temp

/-- Theorem stating that the temperature difference is 8°C -/
theorem wuhan_spring_temp_difference : temp_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_wuhan_spring_temp_difference_l3250_325018


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3250_325099

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (ha_neg : a < 0) (hb_pos : b > 0) :
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3250_325099


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3250_325068

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3250_325068


namespace NUMINAMATH_CALUDE_kylie_stamps_l3250_325090

theorem kylie_stamps (kylie_stamps : ℕ) (nelly_stamps : ℕ) : 
  nelly_stamps = kylie_stamps + 44 →
  kylie_stamps + nelly_stamps = 112 →
  kylie_stamps = 34 := by
sorry

end NUMINAMATH_CALUDE_kylie_stamps_l3250_325090


namespace NUMINAMATH_CALUDE_specific_semicircle_chord_product_l3250_325026

/-- A structure representing a semicircle with equally spaced points -/
structure SemicircleWithPoints where
  radius : ℝ
  num_points : ℕ

/-- The product of chord lengths in a semicircle with equally spaced points -/
def chord_product (s : SemicircleWithPoints) : ℝ :=
  sorry

/-- Theorem stating the product of chord lengths for a specific semicircle configuration -/
theorem specific_semicircle_chord_product :
  let s : SemicircleWithPoints := { radius := 4, num_points := 8 }
  chord_product s = 4718592 := by
  sorry

end NUMINAMATH_CALUDE_specific_semicircle_chord_product_l3250_325026


namespace NUMINAMATH_CALUDE_inequality_proof_l3250_325065

theorem inequality_proof (x b a : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3250_325065


namespace NUMINAMATH_CALUDE_least_perfect_square_exponent_l3250_325091

theorem least_perfect_square_exponent : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (k : ℕ), 2^8 + 2^11 + 2^m = k^2) → m ≥ n) ∧
  (∃ (k : ℕ), 2^8 + 2^11 + 2^n = k^2) ∧
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_least_perfect_square_exponent_l3250_325091


namespace NUMINAMATH_CALUDE_distance_equality_l3250_325000

/-- Given four points in 3D space, prove that a specific point P satisfies the distance conditions --/
theorem distance_equality (A B C D P : ℝ × ℝ × ℝ) : 
  A = (10, 0, 0) →
  B = (0, -6, 0) →
  C = (0, 0, 8) →
  D = (1, 1, 1) →
  P = (3, -2, 5) →
  dist A P = dist B P ∧ 
  dist A P = dist C P ∧ 
  dist A P = dist D P - 3 := by
  sorry

where
  dist : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ
  | (x₁, y₁, z₁), (x₂, y₂, z₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

end NUMINAMATH_CALUDE_distance_equality_l3250_325000


namespace NUMINAMATH_CALUDE_rectangle_area_l3250_325046

/-- Given a rectangle ABCD with the following properties:
  - Sides AB and CD have length 3x
  - Sides AD and BC have length x
  - A circle with radius r is tangent to side AB at its midpoint, AD, and CD
  - 2r = x
  Prove that the area of rectangle ABCD is 12r^2 -/
theorem rectangle_area (x r : ℝ) (h1 : 2 * r = x) : 3 * x * x = 12 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3250_325046


namespace NUMINAMATH_CALUDE_sams_dimes_given_to_dad_l3250_325094

theorem sams_dimes_given_to_dad (initial_dimes : ℕ) (remaining_dimes : ℕ) 
  (h1 : initial_dimes = 9) 
  (h2 : remaining_dimes = 2) : 
  initial_dimes - remaining_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_given_to_dad_l3250_325094


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l3250_325029

theorem max_value_of_product_sum (x y z : ℝ) (h : x + 2*y + z = 7) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (x' y' z' : ℝ), x' + 2*y' + z' = 7 → x'*y' + x'*z' + y'*z' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l3250_325029


namespace NUMINAMATH_CALUDE_quarter_count_proof_l3250_325095

/-- Represents the types of coins in the collection -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a collection of coins -/
structure CoinCollection where
  coins : List Coin

def CoinCollection.averageValue (c : CoinCollection) : ℚ :=
  sorry

def CoinCollection.addDimes (c : CoinCollection) (n : ℕ) : CoinCollection :=
  sorry

def CoinCollection.countQuarters (c : CoinCollection) : ℕ :=
  sorry

theorem quarter_count_proof (c : CoinCollection) :
  c.averageValue = 15 / 100 →
  (c.addDimes 2).averageValue = 17 / 100 →
  c.countQuarters = 4 :=
sorry

end NUMINAMATH_CALUDE_quarter_count_proof_l3250_325095


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3250_325016

def alice_number : ℕ := 36

def is_twice_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2 * p

def has_only_factors_of (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ∣ m

theorem smallest_bob_number :
  ∃ n : ℕ, 
    n > 0 ∧
    is_twice_prime n ∧
    has_only_factors_of n alice_number ∧
    (∀ m : ℕ, m > 0 → is_twice_prime m → has_only_factors_of m alice_number → n ≤ m) ∧
    n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3250_325016


namespace NUMINAMATH_CALUDE_james_huskies_count_l3250_325048

/-- The number of huskies James has -/
def num_huskies : ℕ := sorry

/-- The number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- The number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- The number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- The additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- The difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem james_huskies_count :
  num_huskies = 5 ∧
  num_huskies * pups_per_husky_pitbull +
  num_pitbulls * pups_per_husky_pitbull +
  num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden) =
  num_huskies + num_pitbulls + num_golden_retrievers + pup_adult_difference :=
sorry

end NUMINAMATH_CALUDE_james_huskies_count_l3250_325048


namespace NUMINAMATH_CALUDE_p_necessary_but_not_sufficient_for_q_l3250_325024

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for p to be necessary but not sufficient for q
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- Theorem statement
theorem p_necessary_but_not_sufficient_for_q :
  necessary_but_not_sufficient p q :=
sorry

end NUMINAMATH_CALUDE_p_necessary_but_not_sufficient_for_q_l3250_325024


namespace NUMINAMATH_CALUDE_spurs_total_basketballs_l3250_325002

/-- Represents a basketball team -/
structure BasketballTeam where
  num_players : ℕ
  balls_per_player : ℕ

/-- Calculates the total number of basketballs for a team -/
def total_basketballs (team : BasketballTeam) : ℕ :=
  team.num_players * team.balls_per_player

/-- The Spurs basketball team -/
def spurs : BasketballTeam :=
  { num_players := 35
    balls_per_player := 15 }

/-- Theorem: The Spurs basketball team has 525 basketballs in total -/
theorem spurs_total_basketballs :
  total_basketballs spurs = 525 := by
  sorry

end NUMINAMATH_CALUDE_spurs_total_basketballs_l3250_325002


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3250_325049

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern --/
theorem cistern_wet_surface_area :
  let length : ℝ := 5
  let width : ℝ := 4
  let depth : ℝ := 1.25
  total_wet_surface_area length width depth = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3250_325049


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3250_325093

/-- Given points A(-3,m) and B(-2,n) lying on the hyperbolic function y = (k-1)/x, 
    with m > n, the range of k is k > 1 -/
theorem hyperbola_k_range (k m n : ℝ) : 
  (m = (k - 1) / (-3)) → 
  (n = (k - 1) / (-2)) → 
  (m > n) → 
  (k > 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3250_325093


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_46_l3250_325015

/-- A trapezoid within a rectangle -/
structure TrapezoidInRectangle where
  a : ℝ  -- Length of longer parallel side of trapezoid
  b : ℝ  -- Length of shorter parallel side of trapezoid
  h : ℝ  -- Height of trapezoid (equal to non-parallel sides)
  rect_perimeter : ℝ  -- Perimeter of the rectangle

/-- The perimeter of the trapezoid -/
def trapezoid_perimeter (t : TrapezoidInRectangle) : ℝ :=
  t.a + t.b + 2 * t.h

/-- Theorem stating the perimeter of the trapezoid is 46 meters -/
theorem trapezoid_perimeter_is_46 (t : TrapezoidInRectangle)
  (h1 : t.a = 15)
  (h2 : t.b = 9)
  (h3 : t.rect_perimeter = 52)
  (h4 : t.h = (t.rect_perimeter - 2 * t.a) / 2) :
  trapezoid_perimeter t = 46 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_46_l3250_325015


namespace NUMINAMATH_CALUDE_eulers_formula_applications_l3250_325074

open Complex

theorem eulers_formula_applications :
  let e_2pi_3i : ℂ := Complex.exp ((2 * Real.pi / 3) * I)
  let e_pi_2i : ℂ := Complex.exp ((Real.pi / 2) * I)
  let e_pi_i : ℂ := Complex.exp (Real.pi * I)
  (e_2pi_3i.re < 0 ∧ e_2pi_3i.im > 0) ∧
  (e_pi_2i = I) ∧
  (abs (e_pi_i / (Real.sqrt 3 + I)) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_eulers_formula_applications_l3250_325074


namespace NUMINAMATH_CALUDE_circle_with_AB_diameter_l3250_325008

-- Define the points A and B
def A : ℝ × ℝ := (-3, -5)
def B : ℝ × ℝ := (5, 1)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem circle_with_AB_diameter :
  ∀ x y : ℝ,
  circle_equation x y ↔ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = (1 - t) * A.1 + t * B.1) ∧
     (y = (1 - t) * A.2 + t * B.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_with_AB_diameter_l3250_325008


namespace NUMINAMATH_CALUDE_pauls_shopping_bill_l3250_325047

def dress_shirt_price : ℝ := 15.00
def pants_price : ℝ := 40.00
def suit_price : ℝ := 150.00
def sweater_price : ℝ := 30.00

def num_dress_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.20
def coupon_discount : ℝ := 0.10

def total_before_discount : ℝ := 
  dress_shirt_price * num_dress_shirts +
  pants_price * num_pants +
  suit_price * num_suits +
  sweater_price * num_sweaters

def final_price : ℝ := 
  total_before_discount * (1 - store_discount) * (1 - coupon_discount)

theorem pauls_shopping_bill : final_price = 252.00 := by
  sorry

end NUMINAMATH_CALUDE_pauls_shopping_bill_l3250_325047


namespace NUMINAMATH_CALUDE_min_value_expression_l3250_325064

theorem min_value_expression (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3250_325064


namespace NUMINAMATH_CALUDE_find_m_value_l3250_325005

theorem find_m_value (m : ℤ) : 
  (∃ (x : ℤ), x - m / 3 ≥ 0 ∧ 2 * x - 3 ≥ 3 * (x - 2)) ∧ 
  (∃! (a b : ℤ), a ≠ b ∧ 
    (a - m / 3 ≥ 0 ∧ 2 * a - 3 ≥ 3 * (a - 2)) ∧ 
    (b - m / 3 ≥ 0 ∧ 2 * b - 3 ≥ 3 * (b - 2))) ∧
  (∃ (k : ℤ), k > 0 ∧ 4 * (m + 1) = k * (m^2 - 1)) →
  m = 5 :=
sorry

end NUMINAMATH_CALUDE_find_m_value_l3250_325005


namespace NUMINAMATH_CALUDE_pyramid_height_l3250_325097

/-- The height of a pyramid with a rectangular base and isosceles triangular faces -/
theorem pyramid_height (ab bc : ℝ) (volume : ℝ) (h_ab : ab = 15 * Real.sqrt 3) 
  (h_bc : bc = 14 * Real.sqrt 3) (h_volume : volume = 750) : ℝ := 
  let base_area := ab * bc
  let height := 3 * volume / base_area
  by
    -- Proof goes here
    sorry

#check pyramid_height

end NUMINAMATH_CALUDE_pyramid_height_l3250_325097


namespace NUMINAMATH_CALUDE_remaining_math_problems_l3250_325079

theorem remaining_math_problems (total : ℕ) (completed : ℕ) (remaining : ℕ) : 
  total = 9 → completed = 5 → remaining = total - completed → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_math_problems_l3250_325079


namespace NUMINAMATH_CALUDE_washer_dryer_cost_l3250_325014

theorem washer_dryer_cost (total_cost : ℝ) (price_difference : ℝ) (dryer_cost : ℝ) : 
  total_cost = 1200 →
  price_difference = 220 →
  total_cost = dryer_cost + (dryer_cost + price_difference) →
  dryer_cost = 490 := by
sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_l3250_325014


namespace NUMINAMATH_CALUDE_stating_reach_target_probability_approx_l3250_325041

/-- Represents the probability of winning in a single bet -/
def win_probability : ℝ := 0.1

/-- Represents the cost of a single bet -/
def bet_cost : ℝ := 10

/-- Represents the amount won in a single successful bet -/
def win_amount : ℝ := 30

/-- Represents the initial amount of money -/
def initial_amount : ℝ := 20

/-- Represents the target amount to reach -/
def target_amount : ℝ := 45

/-- 
Represents the probability of reaching the target amount 
starting from the initial amount through a series of bets
-/
noncomputable def reach_target_probability : ℝ := sorry

/-- 
Theorem stating that the probability of reaching the target amount 
is approximately 0.033
-/
theorem reach_target_probability_approx : 
  |reach_target_probability - 0.033| < 0.001 := by sorry

end NUMINAMATH_CALUDE_stating_reach_target_probability_approx_l3250_325041


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l3250_325057

theorem sequence_fourth_term (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = 3^n + 2*n + 1) →
  (∀ n : ℕ, n ≥ 1 → S n = S (n-1) + a n) →
  a 4 = 56 := by
sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l3250_325057


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3250_325033

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3250_325033


namespace NUMINAMATH_CALUDE_cottage_build_time_l3250_325032

/-- Represents the time (in days) it takes to build a cottage -/
def build_time (num_builders : ℕ) (days : ℕ) : Prop :=
  num_builders * days = 24

theorem cottage_build_time :
  build_time 3 8 → build_time 6 4 := by sorry

end NUMINAMATH_CALUDE_cottage_build_time_l3250_325032


namespace NUMINAMATH_CALUDE_smallest_c_value_l3250_325010

/-- Given a function y = a * cos(b * x + c), where a, b, and c are positive constants,
    and the graph reaches its maximum at x = 0, prove that the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3250_325010


namespace NUMINAMATH_CALUDE_pan_division_theorem_main_theorem_l3250_325035

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the side length of a square piece of cake --/
def PieceSize : ℕ := 3

/-- Calculates the number of square pieces that can be cut from a rectangular pan --/
def numberOfPieces (pan : PanDimensions) (pieceSize : ℕ) : ℕ :=
  (pan.length * pan.width) / (pieceSize * pieceSize)

/-- Theorem stating that a 30x24 inch pan can be divided into 80 3-inch square pieces --/
theorem pan_division_theorem (pan : PanDimensions) (h1 : pan.length = 30) (h2 : pan.width = 24) :
  numberOfPieces pan PieceSize = 80 := by
  sorry

/-- Main theorem to be proved --/
theorem main_theorem : ∃ (pan : PanDimensions), 
  pan.length = 30 ∧ pan.width = 24 ∧ numberOfPieces pan PieceSize = 80 := by
  sorry

end NUMINAMATH_CALUDE_pan_division_theorem_main_theorem_l3250_325035


namespace NUMINAMATH_CALUDE_binomial_coefficient_floor_divisibility_l3250_325030

theorem binomial_coefficient_floor_divisibility (p n : ℕ) 
  (hp : Nat.Prime p) (hn : n ≥ p) : 
  (Nat.choose n p - n / p) % p = 0 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_floor_divisibility_l3250_325030


namespace NUMINAMATH_CALUDE_percentage_of_x_l3250_325072

theorem percentage_of_x (x y z : ℚ) : 
  x / y = 4 → 
  x + y = z → 
  y ≠ 0 → 
  z > 0 → 
  (2 * x - y) / x = 175 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l3250_325072


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3250_325017

theorem simplify_and_evaluate (a : ℤ) 
  (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a ≠ 0) (h4 : a ≠ 1) (h5 : a ≠ -1) :
  (a - a^2 / (a^2 - 1)) / (a^2 / (a^2 - 1)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3250_325017


namespace NUMINAMATH_CALUDE_A_eq_B_l3250_325084

/-- A coloring of points in the plane -/
structure Coloring (n : ℕ+) where
  color : ℕ → ℕ → Bool
  valid : ∀ x y x' y', x' ≤ x → y' ≤ y → x + y ≤ n → color x y = false → color x' y' = false

/-- The number of ways to choose n blue points with distinct x-coordinates -/
def A (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The number of ways to choose n blue points with distinct y-coordinates -/
def B (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The main theorem: A = B for any valid coloring -/
theorem A_eq_B (n : ℕ+) (c : Coloring n) : A n c = B n c := by sorry

end NUMINAMATH_CALUDE_A_eq_B_l3250_325084


namespace NUMINAMATH_CALUDE_apples_in_market_l3250_325071

theorem apples_in_market (apples oranges : ℕ) : 
  apples = oranges + 27 →
  apples + oranges = 301 →
  apples = 164 := by
sorry

end NUMINAMATH_CALUDE_apples_in_market_l3250_325071


namespace NUMINAMATH_CALUDE_count_numbers_with_2_and_3_is_52_l3250_325078

/-- A function that counts the number of three-digit numbers with at least one 2 and one 3 -/
def count_numbers_with_2_and_3 : ℕ :=
  let hundreds_not_2_or_3 := 7 * 2  -- Case 1
  let hundreds_is_2 := 10 + 9       -- Case 2
  let hundreds_is_3 := 10 + 9       -- Case 3
  hundreds_not_2_or_3 + hundreds_is_2 + hundreds_is_3

/-- Theorem stating that the count of three-digit numbers with at least one 2 and one 3 is 52 -/
theorem count_numbers_with_2_and_3_is_52 : count_numbers_with_2_and_3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_2_and_3_is_52_l3250_325078


namespace NUMINAMATH_CALUDE_bus_ride_cost_proof_l3250_325042

def bus_ride_cost : ℚ := 1.40
def train_ride_cost : ℚ := bus_ride_cost + 6.85
def combined_cost : ℚ := 9.65
def price_multiple : ℚ := 0.35

theorem bus_ride_cost_proof :
  (train_ride_cost = bus_ride_cost + 6.85) ∧
  (train_ride_cost + bus_ride_cost = combined_cost) ∧
  (∃ n : ℕ, bus_ride_cost = n * price_multiple) ∧
  (∃ m : ℕ, train_ride_cost = m * price_multiple) →
  bus_ride_cost = 1.40 :=
by sorry

end NUMINAMATH_CALUDE_bus_ride_cost_proof_l3250_325042


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3250_325022

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 + 5*x - 24 = 0) ∧
  (∃ x : ℝ, 3*x^2 = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 + 5*x - 24 = 0 ↔ (x = -8 ∨ x = 3)) ∧
  (∀ x : ℝ, 3*x^2 = 2*(2-x) ↔ (x = (-1 + Real.sqrt 13) / 3 ∨ x = (-1 - Real.sqrt 13) / 3)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3250_325022


namespace NUMINAMATH_CALUDE_can_obtain_next_number_l3250_325059

/-- Represents the allowed operations on a number -/
inductive Operation
  | AddNine : Operation
  | DeleteOne : Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (a : ℕ) (ops : List Operation) : ℕ := sorry

/-- Theorem stating that A+1 can always be obtained from A using the allowed operations -/
theorem can_obtain_next_number (A : ℕ) : 
  A > 0 → ∃ (ops : List Operation), applyOperations A ops = A + 1 := by sorry

end NUMINAMATH_CALUDE_can_obtain_next_number_l3250_325059


namespace NUMINAMATH_CALUDE_monica_classes_count_l3250_325050

/-- Represents the number of students in each of Monica's classes -/
def class_sizes : List Nat := [20, 25, 25, 10, 28, 28]

/-- The total number of students Monica sees each day -/
def total_students : Nat := 136

/-- Theorem stating that Monica has 6 classes per day -/
theorem monica_classes_count : List.length class_sizes = 6 ∧ List.sum class_sizes = total_students := by
  sorry

end NUMINAMATH_CALUDE_monica_classes_count_l3250_325050


namespace NUMINAMATH_CALUDE_same_grade_probability_l3250_325045

theorem same_grade_probability (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) 
  (h_total : total = 10)
  (h_first : first = 4)
  (h_second : second = 3)
  (h_third : third = 3)
  (h_sum : first + second + third = total) :
  (Nat.choose first 2 + Nat.choose second 2 + Nat.choose third 2) / Nat.choose total 2 = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_same_grade_probability_l3250_325045


namespace NUMINAMATH_CALUDE_carson_carpool_expense_l3250_325055

/-- Represents the carpool scenario with given parameters --/
structure CarpoolScenario where
  num_friends : Nat
  one_way_miles : Nat
  gas_price : Rat
  miles_per_gallon : Nat
  days_per_week : Nat
  weeks_per_month : Nat

/-- Calculates the monthly gas expense per person for a given carpool scenario --/
def monthly_gas_expense_per_person (scenario : CarpoolScenario) : Rat :=
  let total_miles := 2 * scenario.one_way_miles * scenario.days_per_week * scenario.weeks_per_month
  let total_gallons := total_miles / scenario.miles_per_gallon
  let total_cost := total_gallons * scenario.gas_price
  total_cost / scenario.num_friends

/-- The given carpool scenario --/
def carson_carpool : CarpoolScenario :=
  { num_friends := 5
  , one_way_miles := 21
  , gas_price := 5/2
  , miles_per_gallon := 30
  , days_per_week := 5
  , weeks_per_month := 4
  }

/-- Theorem stating that the monthly gas expense per person for Carson's carpool is $14 --/
theorem carson_carpool_expense :
  monthly_gas_expense_per_person carson_carpool = 14 := by
  sorry


end NUMINAMATH_CALUDE_carson_carpool_expense_l3250_325055


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l3250_325096

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else -(x+1)^2 + 2

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_decreasing_iff_a_range (a : ℝ) :
  (is_decreasing (f a)) ↔ a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l3250_325096


namespace NUMINAMATH_CALUDE_robertos_salary_proof_l3250_325003

theorem robertos_salary_proof (current_salary : ℝ) : 
  current_salary = 134400 →
  ∃ (starting_salary : ℝ),
    starting_salary = 80000 ∧
    current_salary = 1.2 * (1.4 * starting_salary) :=
by
  sorry

end NUMINAMATH_CALUDE_robertos_salary_proof_l3250_325003


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3250_325039

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + Real.sqrt d) + (c - Real.sqrt d) = 6 → 
  (c + Real.sqrt d) * (c - Real.sqrt d) = 4 → 
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3250_325039


namespace NUMINAMATH_CALUDE_problem_solution_l3250_325034

theorem problem_solution (x : ℝ) : (0.20 * x = 0.15 * 1500 - 15) → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3250_325034


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3250_325021

/-- 
A geometric progression with 2 terms, where:
- The last term is 1/3
- The common ratio is 1/3
- The sum of terms is 40/3
Then, the first term is 10.
-/
theorem geometric_progression_first_term 
  (n : ℕ) 
  (last_term : ℚ) 
  (common_ratio : ℚ) 
  (sum : ℚ) : 
  n = 2 ∧ 
  last_term = 1/3 ∧ 
  common_ratio = 1/3 ∧ 
  sum = 40/3 → 
  ∃ (a : ℚ), a = 10 ∧ sum = a * (1 - common_ratio^n) / (1 - common_ratio) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3250_325021


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l3250_325053

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l3250_325053


namespace NUMINAMATH_CALUDE_mike_books_equal_sum_l3250_325006

/-- The number of books Bobby has -/
def bobby_books : Nat := 142

/-- The number of books Kristi has -/
def kristi_books : Nat := 78

/-- The number of books Mike needs to have -/
def mike_books : Nat := bobby_books + kristi_books

theorem mike_books_equal_sum :
  mike_books = bobby_books + kristi_books := by
  sorry

end NUMINAMATH_CALUDE_mike_books_equal_sum_l3250_325006


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_camp_l3250_325075

def lawrence_county_kids_at_home : ℕ := 134867
def outside_county_kids_at_camp : ℕ := 424944
def total_kids_at_camp : ℕ := 458988

theorem lawrence_county_kids_at_camp :
  total_kids_at_camp - outside_county_kids_at_camp = 34044 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_camp_l3250_325075


namespace NUMINAMATH_CALUDE_function_inequality_condition_l3250_325086

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l3250_325086


namespace NUMINAMATH_CALUDE_two_solutions_congruence_l3250_325058

theorem two_solutions_congruence (a : ℕ) (h_a : a < 2007) :
  (∃! u v : ℕ, u < 2007 ∧ v < 2007 ∧ u ≠ v ∧
    (u^2 + a) % 2007 = 0 ∧ (v^2 + a) % 2007 = 0) ↔
  (a % 9 = 0 ∨ a % 9 = 8 ∨ a % 9 = 5 ∨ a % 9 = 2) ∧
  ∃ x : ℕ, x < 223 ∧ (x^2 % 223 = (223 - a % 223) % 223) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_congruence_l3250_325058


namespace NUMINAMATH_CALUDE_cubic_polynomial_conditions_l3250_325013

def f (x : ℚ) : ℚ := 15 * x^3 - 37 * x^2 + 30 * x - 8

theorem cubic_polynomial_conditions :
  f 1 = 0 ∧ f (2/3) = -4 ∧ f (4/5) = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_conditions_l3250_325013


namespace NUMINAMATH_CALUDE_roberto_outfits_l3250_325098

/-- The number of different outfits Roberto can put together -/
def number_of_outfits : ℕ := 180

/-- The number of pairs of trousers Roberto has -/
def number_of_trousers : ℕ := 6

/-- The number of shirts Roberto has -/
def number_of_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def number_of_jackets : ℕ := 4

/-- The number of shirts that cannot be worn with Jacket 1 -/
def number_of_restricted_shirts : ℕ := 2

theorem roberto_outfits :
  number_of_outfits = 
    number_of_trousers * number_of_shirts * number_of_jackets - 
    number_of_trousers * number_of_restricted_shirts := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3250_325098


namespace NUMINAMATH_CALUDE_segment_length_on_number_line_l3250_325001

theorem segment_length_on_number_line : 
  let a : ℝ := -3
  let b : ℝ := 5
  |b - a| = 8 := by sorry

end NUMINAMATH_CALUDE_segment_length_on_number_line_l3250_325001


namespace NUMINAMATH_CALUDE_expression_equals_polynomial_l3250_325081

/-- The given expression is equal to the simplified polynomial for all real x -/
theorem expression_equals_polynomial (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) * (x - 2) -
  (x - 2) * (2 * x^3 - 7 * x^2 + 10) +
  (7 * x - 15) * (x - 2) * (2 * x + 1) =
  x^4 + 23 * x^3 - 78 * x^2 + 39 * x + 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_polynomial_l3250_325081


namespace NUMINAMATH_CALUDE_claire_gift_card_balance_l3250_325085

/-- Calculates the remaining balance on a gift card after a week of purchases --/
def remaining_balance (gift_card_amount : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (cookie_count : ℕ) : ℚ :=
  gift_card_amount - (latte_cost + croissant_cost) * days - cookie_cost * cookie_count

/-- Proves that the remaining balance on Claire's gift card is $43 --/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_claire_gift_card_balance_l3250_325085


namespace NUMINAMATH_CALUDE_function_and_inequality_proof_l3250_325062

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ sorry

-- Define the function g
noncomputable def g : ℝ → ℝ := fun x ↦ (f x - 2 * x) / x

-- Theorem statement
theorem function_and_inequality_proof :
  (∀ x y : ℝ, f (x + y) - f y = (x + 2 * y - 2) * x) ∧
  (f 1 = 0) ∧
  (∀ x : ℝ, f x = (x - 1)^2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → g (2^x) - k * 2^x ≤ 0) ↔ k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_proof_l3250_325062


namespace NUMINAMATH_CALUDE_sanoop_tshirts_l3250_325080

/-- The number of t-shirts Sanoop initially bought -/
def initial_tshirts : ℕ := 8

/-- The initial average price of t-shirts in Rs -/
def initial_avg_price : ℚ := 526

/-- The average price of t-shirts after returning one, in Rs -/
def new_avg_price : ℚ := 505

/-- The price of the returned t-shirt in Rs -/
def returned_price : ℚ := 673

theorem sanoop_tshirts :
  initial_tshirts = 8 ∧
  initial_avg_price * initial_tshirts = 
    new_avg_price * (initial_tshirts - 1) + returned_price :=
by sorry

end NUMINAMATH_CALUDE_sanoop_tshirts_l3250_325080


namespace NUMINAMATH_CALUDE_range_of_a_l3250_325025

theorem range_of_a (a : ℝ) : 
  (∀ x, 0 < x ∧ x < 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) → 
  -1 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3250_325025


namespace NUMINAMATH_CALUDE_meeting_seating_arrangement_l3250_325007

theorem meeting_seating_arrangement (n : ℕ) (h : n = 7) : 
  Nat.choose n 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_meeting_seating_arrangement_l3250_325007


namespace NUMINAMATH_CALUDE_total_cakes_served_l3250_325020

def cakes_served_lunch_today : ℕ := 5
def cakes_served_dinner_today : ℕ := 6
def cakes_served_yesterday : ℕ := 3

theorem total_cakes_served :
  cakes_served_lunch_today + cakes_served_dinner_today + cakes_served_yesterday = 14 :=
by sorry

end NUMINAMATH_CALUDE_total_cakes_served_l3250_325020


namespace NUMINAMATH_CALUDE_equation_solution_l3250_325044

theorem equation_solution :
  ∃ y : ℚ, 3 * y^(1/4) - 5 * (y / y^(3/4)) = 2 + y^(1/4) ∧ y = 16/81 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3250_325044


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3250_325061

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4). -/
theorem y_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 0 → y = 4) :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3250_325061


namespace NUMINAMATH_CALUDE_wolves_out_hunting_l3250_325073

def wolves_in_pack : ℕ := 16
def meat_per_wolf_per_day : ℕ := 8
def days_between_hunts : ℕ := 5
def meat_per_deer : ℕ := 200
def deer_per_hunting_wolf : ℕ := 1

def total_meat_needed : ℕ := wolves_in_pack * meat_per_wolf_per_day * days_between_hunts

def deer_needed : ℕ := (total_meat_needed + meat_per_deer - 1) / meat_per_deer

theorem wolves_out_hunting (hunting_wolves : ℕ) : 
  hunting_wolves * deer_per_hunting_wolf = deer_needed → hunting_wolves = 4 := by
  sorry

end NUMINAMATH_CALUDE_wolves_out_hunting_l3250_325073


namespace NUMINAMATH_CALUDE_matches_played_before_increase_l3250_325012

def cricket_matches (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) : Prop :=
  ∃ m : ℕ,
    (current_average * m + next_match_runs) / (m + 1) = new_average ∧
    m > 0

theorem matches_played_before_increase (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) :
  cricket_matches current_average next_match_runs new_average →
  current_average = 51 →
  next_match_runs = 78 →
  new_average = 54 →
  ∃ m : ℕ, m = 8 ∧ cricket_matches current_average next_match_runs new_average :=
by
  sorry

#check matches_played_before_increase

end NUMINAMATH_CALUDE_matches_played_before_increase_l3250_325012


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3250_325043

theorem complex_fraction_equals_negative_two
  (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a*b + b^2 = 0) :
  (a^7 + b^7) / (a + b)^7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3250_325043


namespace NUMINAMATH_CALUDE_min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l3250_325077

/-- Represents the types of fruits on the magical tree -/
inductive Fruit
  | Banana
  | Orange

/-- Represents the state of the magical tree -/
structure TreeState where
  bananas : Nat
  oranges : Nat

/-- Represents the possible picking actions -/
inductive PickAction
  | PickOne (f : Fruit)
  | PickTwo (f1 f2 : Fruit)

/-- Applies a picking action to the tree state -/
def applyAction (state : TreeState) (action : PickAction) : TreeState :=
  match action with
  | PickAction.PickOne Fruit.Banana => state
  | PickAction.PickOne Fruit.Orange => state
  | PickAction.PickTwo Fruit.Banana Fruit.Banana => 
      { bananas := state.bananas - 2, oranges := state.oranges + 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Banana Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Banana => 
      { bananas := state.bananas, oranges := state.oranges - 1 }

/-- Defines the initial state of the tree -/
def initialState : TreeState := { bananas := 15, oranges := 20 }

/-- Theorem: The minimum number of fruits that can remain on the tree is 1 -/
theorem min_remaining_fruits (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas + 
    (List.foldl applyAction initialState actions).oranges ≥ 1 :=
  sorry

/-- Theorem: The last remaining fruit is always a banana -/
theorem last_fruit_is_banana (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 1 ∧
    (List.foldl applyAction initialState actions).oranges = 0 :=
  sorry

/-- Theorem: It's impossible to remove all fruits from the tree -/
theorem cannot_remove_all_fruits (actions : List PickAction) :
  ¬(∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 0 ∧
    (List.foldl applyAction initialState actions).oranges = 0) :=
  sorry

end NUMINAMATH_CALUDE_min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l3250_325077


namespace NUMINAMATH_CALUDE_power_value_l3250_325036

theorem power_value (m n : ℤ) (x : ℝ) (h1 : x^m = 3) (h2 : x = 2) : x^(2*m+n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_value_l3250_325036


namespace NUMINAMATH_CALUDE_coronavirus_case_ratio_l3250_325083

/-- Proves that the ratio of new coronavirus cases in the second week to the first week is 1/4 -/
theorem coronavirus_case_ratio :
  let first_week : ℕ := 5000
  let third_week (second_week : ℕ) : ℕ := second_week + 2000
  let total_cases : ℕ := 9500
  ∀ second_week : ℕ,
    first_week + second_week + third_week second_week = total_cases →
    (second_week : ℚ) / first_week = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_case_ratio_l3250_325083


namespace NUMINAMATH_CALUDE_sum_opposite_and_abs_l3250_325037

theorem sum_opposite_and_abs : -15 + |(-6)| = -9 := by
  sorry

end NUMINAMATH_CALUDE_sum_opposite_and_abs_l3250_325037


namespace NUMINAMATH_CALUDE_sum_of_critical_slopes_l3250_325069

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The point Q -/
def Q : ℝ × ℝ := (10, 5)

/-- The line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- The quadratic equation representing the intersection of the line and parabola -/
def intersection_quadratic (m : ℝ) (x : ℝ) : ℝ := 
  parabola x - line m x

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 
  m^2 - 4 * (10 * m - 5)

/-- The theorem stating that the sum of the critical slopes is 40 -/
theorem sum_of_critical_slopes : 
  ∃ (r s : ℝ), (∀ m, discriminant m < 0 ↔ r < m ∧ m < s) ∧ r + s = 40 :=
sorry

end NUMINAMATH_CALUDE_sum_of_critical_slopes_l3250_325069


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l3250_325023

theorem ellipse_foci_y_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 5) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = c^2 → 
      (0, c) ∈ {f : ℝ × ℝ | (f.1 - x)^2 + (f.2 - y)^2 + (f.1 + x)^2 + (f.2 - y)^2 = 4 * c^2}) →
  7 < m ∧ m < 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l3250_325023


namespace NUMINAMATH_CALUDE_mom_has_one_eye_l3250_325088

/-- Represents the number of eyes for each family member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  kids_eyes : ℕ
  num_kids : ℕ

/-- The total number of eyes in the monster family -/
def total_eyes (f : MonsterFamily) : ℕ :=
  f.mom_eyes + f.dad_eyes + f.kids_eyes * f.num_kids

/-- Theorem stating that the mom has 1 eye given the conditions -/
theorem mom_has_one_eye (f : MonsterFamily) 
  (h1 : f.dad_eyes = 3)
  (h2 : f.kids_eyes = 4)
  (h3 : f.num_kids = 3)
  (h4 : total_eyes f = 16) : 
  f.mom_eyes = 1 := by
  sorry


end NUMINAMATH_CALUDE_mom_has_one_eye_l3250_325088


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3250_325076

theorem algebraic_expression_value (x : ℝ) : -2 * (2 - x) + (1 + x) = 0 → 2 * x^2 - 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3250_325076


namespace NUMINAMATH_CALUDE_spend_fifty_is_negative_fifty_l3250_325066

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent the sign of a transaction
def transactionSign (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive _ => 1
  | MonetaryTransaction.Spend _ => -1

-- State the theorem
theorem spend_fifty_is_negative_fifty 
  (h1 : transactionSign (MonetaryTransaction.Receive 80) = 1)
  (h2 : transactionSign (MonetaryTransaction.Spend 50) = -transactionSign (MonetaryTransaction.Receive 50)) :
  transactionSign (MonetaryTransaction.Spend 50) * 50 = -50 := by
  sorry

end NUMINAMATH_CALUDE_spend_fifty_is_negative_fifty_l3250_325066


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l3250_325004

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (h1 : original_intensity = 0.6) 
  (h2 : added_intensity = 0.3) 
  (h3 : replaced_fraction = 2/3) : 
  (1 - replaced_fraction) * original_intensity + replaced_fraction * added_intensity = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_intensity_l3250_325004


namespace NUMINAMATH_CALUDE_broccoli_production_increase_l3250_325051

theorem broccoli_production_increase :
  ∀ (last_year_side : ℕ) (this_year_side : ℕ),
    last_year_side = 50 →
    this_year_side = 51 →
    this_year_side * this_year_side - last_year_side * last_year_side = 101 :=
by sorry

end NUMINAMATH_CALUDE_broccoli_production_increase_l3250_325051


namespace NUMINAMATH_CALUDE_diophantine_equation_only_zero_solution_l3250_325028

theorem diophantine_equation_only_zero_solution (x y u t : ℤ) 
  (h : x^2 + y^2 = 1974 * (u^2 + t^2)) : x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_only_zero_solution_l3250_325028


namespace NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_for_not_p_true_l3250_325082

theorem either_false_sufficient_not_necessary_for_not_p_true (p q : Prop) :
  (((¬p ∧ ¬q) → ¬p) ∧ ∃ (r : Prop), (¬r ∧ ¬(¬r ∧ ¬q))) := by
  sorry

end NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_for_not_p_true_l3250_325082


namespace NUMINAMATH_CALUDE_nails_per_station_l3250_325056

theorem nails_per_station (total_nails : ℕ) (num_stations : ℕ) 
  (h1 : total_nails = 140) (h2 : num_stations = 20) :
  total_nails / num_stations = 7 := by
  sorry

end NUMINAMATH_CALUDE_nails_per_station_l3250_325056


namespace NUMINAMATH_CALUDE_days_to_finish_book_l3250_325087

theorem days_to_finish_book (total_pages book_chapters pages_per_day : ℕ) : 
  total_pages = 193 → book_chapters = 15 → pages_per_day = 44 → 
  (total_pages + pages_per_day - 1) / pages_per_day = 5 := by
sorry

end NUMINAMATH_CALUDE_days_to_finish_book_l3250_325087


namespace NUMINAMATH_CALUDE_x_axis_fixed_slope_two_invariant_l3250_325052

/-- Transformation that maps a point (x, y) to (x-y, -y) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - p.2, -p.2)

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

theorem x_axis_fixed :
  ∀ (x : ℝ), transform (x, 0) = (x, 0) := by sorry

theorem slope_two_invariant (b : ℝ) :
  ∀ (x y : ℝ), 
    (Line.contains { slope := 2, intercept := b } (x, y)) →
    (Line.contains { slope := 2, intercept := b } (transform (x, y))) := by sorry

end NUMINAMATH_CALUDE_x_axis_fixed_slope_two_invariant_l3250_325052


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3250_325089

theorem repeating_decimal_sum : 
  let x : ℚ := (23 : ℚ) / 99
  let y : ℚ := (14 : ℚ) / 999
  let z : ℚ := (6 : ℚ) / 9999
  x + y + z = (2469 : ℚ) / 9999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3250_325089


namespace NUMINAMATH_CALUDE_house_to_library_distance_l3250_325038

/-- Represents the distances between locations in miles -/
structure Distances where
  total : ℝ
  library_to_post_office : ℝ
  post_office_to_home : ℝ

/-- Calculates the distance from house to library -/
def distance_house_to_library (d : Distances) : ℝ :=
  d.total - d.library_to_post_office - d.post_office_to_home

/-- Theorem stating the distance from house to library is 0.3 miles -/
theorem house_to_library_distance (d : Distances) 
  (h1 : d.total = 0.8)
  (h2 : d.library_to_post_office = 0.1)
  (h3 : d.post_office_to_home = 0.4) : 
  distance_house_to_library d = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_house_to_library_distance_l3250_325038
