import Mathlib

namespace NUMINAMATH_CALUDE_unknown_number_proof_l3757_375746

theorem unknown_number_proof (a b : ℝ) : 
  (a - 3 = b - a) →  -- arithmetic sequence condition
  ((a - 6) / 3 = b / (a - 6)) →  -- geometric sequence condition
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3757_375746


namespace NUMINAMATH_CALUDE_two_point_distribution_a_value_l3757_375712

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  a : ℝ
  prob_zero : ℝ := 2 * a^2
  prob_one : ℝ := a

/-- The sum of probabilities in a two-point distribution equals 1 -/
axiom prob_sum_eq_one (X : TwoPointDistribution) : X.prob_zero + X.prob_one = 1

/-- Theorem: The value of 'a' in the two-point distribution is 1/2 -/
theorem two_point_distribution_a_value (X : TwoPointDistribution) : X.a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_a_value_l3757_375712


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3757_375758

theorem quadratic_function_property (b c m n : ℝ) :
  let f := fun (x : ℝ) => x^2 + b*x + c
  (f m = n ∧ f (m + 1) = n) →
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 2 → f x₁ > f x₂) →
  m ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3757_375758


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l3757_375768

/-- Represents a parabola in the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Our specific parabola -/
def our_parabola : Parabola := { a := -3, h := 1, k := 2 }

/-- Theorem: The vertex of our parabola is (1,2) -/
theorem vertex_of_our_parabola : vertex our_parabola = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l3757_375768


namespace NUMINAMATH_CALUDE_shanghai_masters_matches_l3757_375776

/-- Represents the tournament structure described in the problem -/
structure Tournament :=
  (num_players : Nat)
  (num_groups : Nat)
  (players_per_group : Nat)
  (advancing_per_group : Nat)

/-- Calculates the number of matches in a round-robin tournament -/
def round_robin_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  let group_matches := t.num_groups * round_robin_matches t.players_per_group
  let elimination_matches := t.num_groups * t.advancing_per_group / 2
  let final_matches := 2
  group_matches + elimination_matches + final_matches

/-- Theorem stating that the total number of matches in the given tournament format is 16 -/
theorem shanghai_masters_matches :
  ∃ t : Tournament, t.num_players = 8 ∧ t.num_groups = 2 ∧ t.players_per_group = 4 ∧ t.advancing_per_group = 2 ∧ total_matches t = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_shanghai_masters_matches_l3757_375776


namespace NUMINAMATH_CALUDE_necklace_beads_l3757_375749

theorem necklace_beads (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) (silver : ℕ) :
  total = 40 →
  red = 2 * blue →
  white = blue + red →
  silver = 10 →
  blue + red + white + silver = total →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l3757_375749


namespace NUMINAMATH_CALUDE_smallest_product_factors_l3757_375777

/-- A structure representing an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ -- first term
  d : ℕ -- common difference

/-- A structure representing a geometric sequence -/
structure GeometricSequence where
  b : ℕ -- first term
  r : ℕ -- common ratio

/-- The product of the first four terms of an arithmetic sequence -/
def arithProduct (seq : ArithmeticSequence) : ℕ :=
  seq.a * (seq.a + seq.d) * (seq.a + 2*seq.d) * (seq.a + 3*seq.d)

/-- The product of the first four terms of a geometric sequence -/
def geoProduct (seq : GeometricSequence) : ℕ :=
  seq.b * (seq.b * seq.r) * (seq.b * seq.r^2) * (seq.b * seq.r^3)

/-- The number of positive factors of a natural number -/
def numPositiveFactors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem smallest_product_factors : 
  ∃ (n : ℕ) (arith : ArithmeticSequence) (geo : GeometricSequence), 
    n > 500000 ∧ 
    n = arithProduct arith ∧ 
    n = geoProduct geo ∧
    (∀ m, m > 500000 → m = arithProduct arith → m = geoProduct geo → m ≥ n) ∧
    numPositiveFactors n = 56 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_factors_l3757_375777


namespace NUMINAMATH_CALUDE_remainder_property_l3757_375795

theorem remainder_property (x : ℕ) (h : x > 0) :
  (100 % x = 4) → ((100 + x) % x = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l3757_375795


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l3757_375748

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l3757_375748


namespace NUMINAMATH_CALUDE_total_price_is_1797_80_l3757_375793

/-- The price of Marion's bike in dollars -/
def marion_bike_price : ℝ := 356

/-- The price of Stephanie's bike before discount in dollars -/
def stephanie_bike_price_before_discount : ℝ := 2 * marion_bike_price

/-- The discount percentage Stephanie received -/
def stephanie_discount_percent : ℝ := 0.1

/-- The price of Patrick's bike before promotion in dollars -/
def patrick_bike_price_before_promotion : ℝ := 3 * marion_bike_price

/-- The percentage of the original price Patrick pays -/
def patrick_payment_percent : ℝ := 0.75

/-- The total price paid for all three bikes -/
def total_price : ℝ := 
  marion_bike_price + 
  stephanie_bike_price_before_discount * (1 - stephanie_discount_percent) + 
  patrick_bike_price_before_promotion * patrick_payment_percent

/-- Theorem stating that the total price paid for the three bikes is $1797.80 -/
theorem total_price_is_1797_80 : total_price = 1797.80 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_1797_80_l3757_375793


namespace NUMINAMATH_CALUDE_smallest_n_both_composite_l3757_375762

def is_composite (n : ℕ) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_n_both_composite :
  (∀ n : ℕ, n > 0 ∧ n < 13 → ¬(is_composite (2*n - 1) ∧ is_composite (2*n + 1))) ∧
  (is_composite (2*13 - 1) ∧ is_composite (2*13 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_both_composite_l3757_375762


namespace NUMINAMATH_CALUDE_min_value_on_interval_l3757_375703

/-- The function f(x) = -x³ + 3x² + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a y ≤ f a x) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l3757_375703


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3757_375729

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 → 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧ 
    x = 2^(Nat.log 2 x) * 7 * p * q) →
  x = 728 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3757_375729


namespace NUMINAMATH_CALUDE_turtle_count_l3757_375751

theorem turtle_count (T : ℕ) : 
  (T + (3 * T - 2)) / 2 = 17 → T = 9 := by
  sorry

end NUMINAMATH_CALUDE_turtle_count_l3757_375751


namespace NUMINAMATH_CALUDE_no_solution_to_system_l3757_375718

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧
    (x^2 + x*y + 8*z^2 = 100) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l3757_375718


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3757_375794

/-- Represents a sampling method -/
structure SamplingMethod where
  -- Add necessary fields here
  reasonable : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Represents the probability of an individual being selected in a sample -/
def selectionProbability (s : Sample) (individual : ℕ) : ℝ :=
  -- Definition would go here
  sorry

theorem equal_selection_probability 
  (s1 s2 : Sample) 
  (h1 : s1.size = s2.size) 
  (h2 : s1.method.reasonable) 
  (h3 : s2.method.reasonable) 
  (individual : ℕ) : 
  selectionProbability s1 individual = selectionProbability s2 individual :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l3757_375794


namespace NUMINAMATH_CALUDE_larger_number_problem_l3757_375743

theorem larger_number_problem (x y : ℕ) : 
  x + y = 64 → y = x + 12 → y = 38 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3757_375743


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3757_375784

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3757_375784


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3757_375787

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3757_375787


namespace NUMINAMATH_CALUDE_annas_cupcake_sales_l3757_375767

/-- Anna's cupcake sales problem -/
theorem annas_cupcake_sales (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℕ) (sold_fraction : ℚ) : 
  num_trays = 4 →
  cupcakes_per_tray = 20 →
  price_per_cupcake = 2 →
  sold_fraction = 3 / 5 →
  (num_trays * cupcakes_per_tray * sold_fraction * price_per_cupcake : ℚ) = 96 :=
by sorry

end NUMINAMATH_CALUDE_annas_cupcake_sales_l3757_375767


namespace NUMINAMATH_CALUDE_only_setC_is_pythagorean_triple_l3757_375750

-- Define the sets of numbers
def setA : List ℕ := [1, 2, 2]
def setB : List ℕ := [3^2, 4^2, 5^2]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 6, 6]

-- Define a function to check if a list of three numbers forms a Pythagorean triple
def isPythagoreanTriple (triple : List ℕ) : Prop :=
  match triple with
  | [a, b, c] => a^2 + b^2 = c^2
  | _ => False

-- Theorem stating that only setC forms a Pythagorean triple
theorem only_setC_is_pythagorean_triple :
  ¬(isPythagoreanTriple setA) ∧
  ¬(isPythagoreanTriple setB) ∧
  (isPythagoreanTriple setC) ∧
  ¬(isPythagoreanTriple setD) :=
sorry

end NUMINAMATH_CALUDE_only_setC_is_pythagorean_triple_l3757_375750


namespace NUMINAMATH_CALUDE_popsicle_sticks_remaining_l3757_375726

theorem popsicle_sticks_remaining (initial : Real) (given_away : Real) :
  initial = 63.0 →
  given_away = 50.0 →
  initial - given_away = 13.0 := by sorry

end NUMINAMATH_CALUDE_popsicle_sticks_remaining_l3757_375726


namespace NUMINAMATH_CALUDE_removed_sector_angle_l3757_375759

/-- Given a circular piece of paper with radius 15 cm, if a cone is formed from the remaining sector
    after removing a part, and this cone has a radius of 10 cm and a volume of 500π cm³,
    then the angle measure of the removed sector is 120°. -/
theorem removed_sector_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 15 →
  cone_radius = 10 →
  cone_volume = 500 * Real.pi →
  ∃ (removed_angle : ℝ), removed_angle = 120 ∧ 0 ≤ removed_angle ∧ removed_angle ≤ 360 :=
by sorry

end NUMINAMATH_CALUDE_removed_sector_angle_l3757_375759


namespace NUMINAMATH_CALUDE_remaining_distance_l3757_375707

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 384) :
  total_distance - driven_distance = 816 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l3757_375707


namespace NUMINAMATH_CALUDE_books_per_continent_l3757_375735

theorem books_per_continent 
  (total_books : ℕ) 
  (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end NUMINAMATH_CALUDE_books_per_continent_l3757_375735


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3757_375766

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3757_375766


namespace NUMINAMATH_CALUDE_fish_purchase_total_l3757_375754

theorem fish_purchase_total (yesterday_fish : ℕ) (yesterday_cost : ℕ) (today_extra_cost : ℕ) : 
  yesterday_fish = 10 →
  yesterday_cost = 3000 →
  today_extra_cost = 6000 →
  ∃ (today_fish : ℕ), 
    (yesterday_fish + today_fish = 40 ∧ 
     yesterday_cost + today_extra_cost = (yesterday_cost / yesterday_fish) * (yesterday_fish + today_fish)) := by
  sorry

#check fish_purchase_total

end NUMINAMATH_CALUDE_fish_purchase_total_l3757_375754


namespace NUMINAMATH_CALUDE_toy_car_production_l3757_375714

theorem toy_car_production (yesterday : ℕ) (today : ℕ) (total : ℕ) : 
  today = 2 * yesterday → 
  total = yesterday + today → 
  total = 180 → 
  yesterday = 60 :=
by sorry

end NUMINAMATH_CALUDE_toy_car_production_l3757_375714


namespace NUMINAMATH_CALUDE_exist_same_color_square_product_l3757_375756

/-- A coloring of natural numbers using two colors. -/
def TwoColoring := ℕ → Bool

/-- Theorem stating that for any two-coloring of natural numbers,
    there exist A, B, and C of the same color such that A * B = C * C. -/
theorem exist_same_color_square_product (coloring : TwoColoring) :
  ∃ (A B C : ℕ), coloring A = coloring B ∧ coloring B = coloring C ∧ A * B = C * C := by
  sorry

end NUMINAMATH_CALUDE_exist_same_color_square_product_l3757_375756


namespace NUMINAMATH_CALUDE_expression_value_l3757_375706

theorem expression_value (x y z : ℚ) 
  (eq1 : 2 * x - y = 4)
  (eq2 : 3 * x + z = 7)
  (eq3 : y = 2 * z) :
  6 * x - 3 * y + 3 * z = 51 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3757_375706


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3757_375733

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^4) (h2 : units_digit m = 6) : 
  units_digit n = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3757_375733


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3757_375732

theorem absolute_value_sum (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a * b < 0) → abs (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3757_375732


namespace NUMINAMATH_CALUDE_tommys_tomato_profit_l3757_375734

/-- Represents the problem of calculating Tommy's profit from selling tomatoes -/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20 -- kg
  let num_crates : ℕ := 3
  let crates_cost : ℕ := 330 -- $
  let selling_price : ℕ := 6 -- $ per kg
  let rotten_tomatoes : ℕ := 3 -- kg
  
  let total_capacity : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_capacity - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - crates_cost

  profit = 12 := by
  sorry

/- Note: We use ℕ (natural numbers) for non-negative integers and ℤ (integers) for the final profit calculation to allow for the possibility of negative profit. -/

end NUMINAMATH_CALUDE_tommys_tomato_profit_l3757_375734


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_l3757_375745

open BigOperators

theorem sum_reciprocal_product : ∑ n in Finset.range 6, 1 / ((n + 3) * (n + 4)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_l3757_375745


namespace NUMINAMATH_CALUDE_prime_between_n_and_nfactorial_l3757_375785

theorem prime_between_n_and_nfactorial (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n < p ∧ p ≤ n! :=
by sorry

end NUMINAMATH_CALUDE_prime_between_n_and_nfactorial_l3757_375785


namespace NUMINAMATH_CALUDE_impossible_sum_16_l3757_375782

def standard_die := Finset.range 6

theorem impossible_sum_16 (roll1 roll2 : ℕ) :
  roll1 ∈ standard_die → roll2 ∈ standard_die → roll1 + roll2 ≠ 16 := by
  sorry

end NUMINAMATH_CALUDE_impossible_sum_16_l3757_375782


namespace NUMINAMATH_CALUDE_subtract_product_equality_l3757_375730

theorem subtract_product_equality : 7899665 - 12 * 3 * 2 = 7899593 := by
  sorry

end NUMINAMATH_CALUDE_subtract_product_equality_l3757_375730


namespace NUMINAMATH_CALUDE_investment_doubling_time_l3757_375781

/-- The minimum number of years required for an investment to at least double -/
theorem investment_doubling_time (A r : ℝ) (h1 : A > 0) (h2 : r > 0) :
  let t := Real.log 2 / Real.log (1 + r)
  ∀ s : ℝ, s ≥ t → A * (1 + r) ^ s ≥ 2 * A :=
by sorry

end NUMINAMATH_CALUDE_investment_doubling_time_l3757_375781


namespace NUMINAMATH_CALUDE_y_intercept_not_z_l3757_375716

/-- For a line ax + by - z = 0 where b ≠ 0, the y-intercept is not equal to z -/
theorem y_intercept_not_z (a b z : ℝ) (h : b ≠ 0) :
  ∃ (y_intercept : ℝ), y_intercept = z / b ∧ y_intercept ≠ z := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_not_z_l3757_375716


namespace NUMINAMATH_CALUDE_chord_distance_half_arc_l3757_375741

/-- Given a circle with radius R and a chord at distance d from the center,
    the distance of the chord corresponding to an arc half as long is √(R(R+d)/2). -/
theorem chord_distance_half_arc (R d : ℝ) (h₁ : R > 0) (h₂ : 0 ≤ d) (h₃ : d < R) :
  let distance_half_arc := Real.sqrt (R * (R + d) / 2)
  distance_half_arc > 0 ∧ distance_half_arc < R :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_half_arc_l3757_375741


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l3757_375799

theorem trigonometric_system_solution (x y : ℝ) (k : ℤ) :
  (Real.cos x)^2 + (Real.cos y)^2 = 0.25 →
  x + y = 5 * Real.pi / 6 →
  ((x = Real.pi / 2 * (2 * ↑k + 1) ∧ y = Real.pi / 3 * (1 - 3 * ↑k)) ∨
   (x = Real.pi / 3 * (3 * ↑k + 1) ∧ y = Real.pi / 2 * (1 - 2 * ↑k))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l3757_375799


namespace NUMINAMATH_CALUDE_rectangle_area_l3757_375704

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  width * length = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3757_375704


namespace NUMINAMATH_CALUDE_rectangle_area_l3757_375723

theorem rectangle_area (k : ℕ+) : 
  let square_side : ℝ := (16 : ℝ).sqrt
  let rectangle_length : ℝ := k * square_side
  let rectangle_breadth : ℝ := 11
  rectangle_length * rectangle_breadth = 220 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3757_375723


namespace NUMINAMATH_CALUDE_tan_equality_implies_sixty_degrees_l3757_375773

theorem tan_equality_implies_sixty_degrees (n : ℤ) :
  -90 < n ∧ n < 90 →
  Real.tan (n * π / 180) = Real.tan (240 * π / 180) →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_sixty_degrees_l3757_375773


namespace NUMINAMATH_CALUDE_A_inter_B_eq_open_interval_l3757_375792

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

-- State the theorem
theorem A_inter_B_eq_open_interval : A ∩ B = {x | 0 < x ∧ x < Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_open_interval_l3757_375792


namespace NUMINAMATH_CALUDE_total_books_read_is_48cs_l3757_375713

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  c * s * 4 * 12

/-- Theorem: The total number of books read by the entire student body in one year is 48cs -/
theorem total_books_read_is_48cs (c s : ℕ) : total_books_read c s = 48 * c * s := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_48cs_l3757_375713


namespace NUMINAMATH_CALUDE_anne_cleaning_rate_l3757_375765

-- Define cleaning rates for Bruce, Anne, and Carl
variable (B A C : ℚ)

-- Define the conditions
def condition1 : Prop := B + A + C = 1/6
def condition2 : Prop := B + 2*A + 3*C = 1/2
def condition3 : Prop := B + A = 1/4
def condition4 : Prop := B + C = 1/3

-- Theorem to prove
theorem anne_cleaning_rate 
  (h1 : condition1 B A C)
  (h2 : condition2 B A C)
  (h3 : condition3 B A)
  (h4 : condition4 B C) :
  A = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_anne_cleaning_rate_l3757_375765


namespace NUMINAMATH_CALUDE_non_union_women_percentage_l3757_375702

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℝ
  men : ℝ
  unionized : ℝ
  unionized_men : ℝ

/-- Conditions given in the problem -/
def company_conditions (c : CompanyEmployees) : Prop :=
  c.men / c.total = 0.54 ∧
  c.unionized / c.total = 0.6 ∧
  c.unionized_men / c.unionized = 0.7

/-- The theorem to be proved -/
theorem non_union_women_percentage (c : CompanyEmployees) 
  (h : company_conditions c) : 
  (c.total - c.unionized - (c.men - c.unionized_men)) / (c.total - c.unionized) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_non_union_women_percentage_l3757_375702


namespace NUMINAMATH_CALUDE_system_solvable_l3757_375761

/-- The system of equations has a real solution if and only if m ≠ 3/2 -/
theorem system_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3/2 := by
sorry

end NUMINAMATH_CALUDE_system_solvable_l3757_375761


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l3757_375708

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l3757_375708


namespace NUMINAMATH_CALUDE_exists_all_strawberry_day_l3757_375783

-- Define the type for our matrix
def WorkSchedule := Matrix (Fin 7) (Fin 16) Bool

-- Define the conditions
def first_day_all_mine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule i 0 = false

def at_least_three_different (schedule : WorkSchedule) : Prop :=
  ∀ j k : Fin 16, j ≠ k → 
    (∃ (s : Finset (Fin 7)), s.card ≥ 3 ∧ 
      (∀ i ∈ s, schedule i j ≠ schedule i k))

-- The main theorem
theorem exists_all_strawberry_day (schedule : WorkSchedule) 
  (h1 : first_day_all_mine schedule)
  (h2 : at_least_three_different schedule) : 
  ∃ j : Fin 16, ∀ i : Fin 7, schedule i j = true :=
sorry

end NUMINAMATH_CALUDE_exists_all_strawberry_day_l3757_375783


namespace NUMINAMATH_CALUDE_fixed_fee_determination_l3757_375738

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixedFee : ℝ
  hourlyCharge : ℝ

/-- Calculates the total bill given the billing system and hours used -/
def calculateBill (bs : BillingSystem) (hours : ℝ) : ℝ :=
  bs.fixedFee + bs.hourlyCharge * hours

theorem fixed_fee_determination (bs : BillingSystem) 
  (h1 : calculateBill bs 1 = 18.70)
  (h2 : calculateBill bs 3 = 34.10) : 
  bs.fixedFee = 11.00 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_determination_l3757_375738


namespace NUMINAMATH_CALUDE_average_equation_l3757_375737

theorem average_equation (x y : ℚ) : 
  x = 50 / 11399 ∧ y = -11275 / 151 →
  (List.sum (List.range 150) + x + y) / 152 = 75 * x + y := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l3757_375737


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3757_375705

def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 11 * x + 5 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference (p q : ℕ) : 
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    |x₁ - x₂| = Real.sqrt p / q) →
  q > 0 →
  is_square_free p →
  p + q = 83 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3757_375705


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l3757_375752

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem quarter_of_fifth_of_sixth_of_120 :
  (1 / 4 : ℚ) * ((1 / 5 : ℚ) * ((1 / 6 : ℚ) * 120)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l3757_375752


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3757_375775

def p : ℕ := (List.range 35).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ∧ 3^k ∣ p ∧ ∀ m > k, ¬(3^m ∣ p) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3757_375775


namespace NUMINAMATH_CALUDE_quarters_borrowed_l3757_375764

/-- Represents the number of quarters Jessica had initially -/
def initial_quarters : ℕ := 8

/-- Represents the number of quarters Jessica has now -/
def current_quarters : ℕ := 5

/-- Represents the number of quarters Jessica's sister borrowed -/
def borrowed_quarters : ℕ := initial_quarters - current_quarters

theorem quarters_borrowed :
  borrowed_quarters = initial_quarters - current_quarters :=
by sorry

end NUMINAMATH_CALUDE_quarters_borrowed_l3757_375764


namespace NUMINAMATH_CALUDE_geometric_progression_floor_sum_l3757_375780

theorem geometric_progression_floor_sum (a b c k r : ℝ) : 
  a > 0 → b > 0 → c > 0 → k > 0 → r > 1 → 
  b = k * r → c = k * r^2 →
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_floor_sum_l3757_375780


namespace NUMINAMATH_CALUDE_square_equation_solution_l3757_375760

theorem square_equation_solution (x y : ℝ) 
  (h1 : x^2 = y + 3) 
  (h2 : x = 6) : 
  y = 33 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3757_375760


namespace NUMINAMATH_CALUDE_inequality_sum_l3757_375736

theorem inequality_sum (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l3757_375736


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3757_375740

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = (1 - Complex.I)^2 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3757_375740


namespace NUMINAMATH_CALUDE_sequences_theorem_l3757_375719

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def geometric_sequence (b : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem sequences_theorem (a b : ℕ → ℚ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 2 →
  b 2 + b 3 = 12 →
  b 3 = a 4 - 2 * a 1 →
  sum_arithmetic a 11 = 11 * b 4 →
  (∀ n : ℕ, n > 0 → a n = 3 * n - 2) ∧
  (∀ n : ℕ, n > 0 → b n = 2^n) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => a (2 * (i + 1)) * b (2 * i + 1)) = 
      (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sequences_theorem_l3757_375719


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_l3757_375722

/-- Two lines intersecting at a point implies a specific sum of their slopes and y-intercepts -/
theorem intersecting_lines_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 2 → y = 4 * x + b → x = 4 ∧ y = 8) → 
  b + m = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_sum_l3757_375722


namespace NUMINAMATH_CALUDE_alfonso_work_weeks_l3757_375796

def hourly_rate : ℝ := 6
def monday_hours : ℝ := 2
def tuesday_hours : ℝ := 3
def wednesday_hours : ℝ := 4
def thursday_hours : ℝ := 2
def friday_hours : ℝ := 3
def helmet_cost : ℝ := 340
def gloves_cost : ℝ := 45
def current_savings : ℝ := 40
def miscellaneous_expenses : ℝ := 20

def weekly_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def weekly_earnings : ℝ := weekly_hours * hourly_rate
def total_cost : ℝ := helmet_cost + gloves_cost + miscellaneous_expenses
def additional_earnings_needed : ℝ := total_cost - current_savings

theorem alfonso_work_weeks : 
  ∃ n : ℕ, n * weekly_earnings ≥ additional_earnings_needed ∧ 
           (n - 1) * weekly_earnings < additional_earnings_needed ∧
           n = 5 :=
sorry

end NUMINAMATH_CALUDE_alfonso_work_weeks_l3757_375796


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3757_375757

theorem cube_equation_solution (x : ℝ) : (x + 3)^3 = -64 → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3757_375757


namespace NUMINAMATH_CALUDE_vegan_meal_clients_l3757_375753

theorem vegan_meal_clients (total : ℕ) (kosher : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 ∧ kosher = 8 ∧ both = 3 ∧ neither = 18 →
  ∃ vegan : ℕ, vegan = 10 ∧ vegan + (kosher - both) + neither = total :=
by sorry

end NUMINAMATH_CALUDE_vegan_meal_clients_l3757_375753


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l3757_375786

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℝ := (meters_to_cm) ^ 3

theorem cubic_meter_to_cubic_cm : 
  cubic_cm_in_cubic_meter = 1000000 :=
sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l3757_375786


namespace NUMINAMATH_CALUDE_income_calculation_l3757_375744

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 10 →  -- ratio of income to expenditure is 10:4
  savings = income - expenditure →  -- savings definition
  savings = 11400 →  -- given savings amount
  income = 19000 := by  -- prove that income is 19000
sorry

end NUMINAMATH_CALUDE_income_calculation_l3757_375744


namespace NUMINAMATH_CALUDE_cubic_polynomial_evaluation_l3757_375747

theorem cubic_polynomial_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_evaluation_l3757_375747


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3757_375771

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + m = 0) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3757_375771


namespace NUMINAMATH_CALUDE_sum_of_radii_l3757_375715

/-- The sum of radii of circles tangent to x and y axes and externally tangent to a circle at (5,0) with radius 1.5 -/
theorem sum_of_radii : ∃ (r₁ r₂ : ℝ),
  r₁ > 0 ∧ r₂ > 0 ∧
  (r₁ - 5)^2 + r₁^2 = (r₁ + 1.5)^2 ∧
  (r₂ - 5)^2 + r₂^2 = (r₂ + 1.5)^2 ∧
  r₁ + r₂ = 13 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_radii_l3757_375715


namespace NUMINAMATH_CALUDE_fraction_calculation_l3757_375763

theorem fraction_calculation : (1/4 + 1/6 - 1/2) / (-1/24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3757_375763


namespace NUMINAMATH_CALUDE_sixth_result_l3757_375711

theorem sixth_result (total_results : ℕ) (all_average first_six_average last_six_average : ℚ) :
  total_results = 11 →
  all_average = 52 →
  first_six_average = 49 →
  last_six_average = 52 →
  ∃ (sixth_result : ℚ),
    sixth_result = 34 ∧
    (6 * first_six_average - sixth_result) + sixth_result + (6 * last_six_average - sixth_result) = total_results * all_average :=
by sorry

end NUMINAMATH_CALUDE_sixth_result_l3757_375711


namespace NUMINAMATH_CALUDE_flower_percentage_l3757_375779

theorem flower_percentage (total_flowers : ℕ) (yellow_flowers : ℕ) (purple_increase : ℚ) :
  total_flowers = 35 →
  yellow_flowers = 10 →
  purple_increase = 80 / 100 →
  let purple_flowers := yellow_flowers + (purple_increase * yellow_flowers).floor
  let green_flowers := total_flowers - yellow_flowers - purple_flowers
  let yellow_and_purple := yellow_flowers + purple_flowers
  (green_flowers : ℚ) / yellow_and_purple * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_flower_percentage_l3757_375779


namespace NUMINAMATH_CALUDE_als_investment_l3757_375789

theorem als_investment (al betty clare : ℝ) 
  (total_initial : al + betty + clare = 1200)
  (total_final : (al - 200) + (2 * betty) + (1.5 * clare) = 1800)
  : al = 600 := by
  sorry

end NUMINAMATH_CALUDE_als_investment_l3757_375789


namespace NUMINAMATH_CALUDE_eagles_winning_percentage_min_additional_games_is_minimum_l3757_375774

/-- The minimum number of additional games needed for the Eagles to win at least 90% of all games -/
def min_additional_games : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won by the Eagles -/
def initial_eagles_wins : ℕ := 1

/-- The minimum winning percentage required for the Eagles -/
def min_winning_percentage : ℚ := 9/10

theorem eagles_winning_percentage (M : ℕ) :
  (initial_eagles_wins + M : ℚ) / (initial_games + M) ≥ min_winning_percentage ↔ M ≥ min_additional_games :=
sorry

theorem min_additional_games_is_minimum :
  ∀ M : ℕ, M < min_additional_games →
    (initial_eagles_wins + M : ℚ) / (initial_games + M) < min_winning_percentage :=
sorry

end NUMINAMATH_CALUDE_eagles_winning_percentage_min_additional_games_is_minimum_l3757_375774


namespace NUMINAMATH_CALUDE_max_ab_linear_function_l3757_375742

/-- Given a linear function f(x) = ax + b where a and b are real numbers,
    if |f(x)| ≤ 1 for all x in [0, 1], then the maximum value of ab is 1/4. -/
theorem max_ab_linear_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) →
  ab ≤ (1 : ℝ) / 4 ∧ ∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_linear_function_l3757_375742


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3757_375798

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 2, 4}
def B : Set Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3757_375798


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l3757_375727

/-- Two integers are consecutive even integers -/
def ConsecutiveEvenIntegers (p q : ℤ) : Prop :=
  ∃ (k : ℤ), p = 2 * k ∧ q = 2 * k + 2

/-- The expression D for two integers -/
def D (p q : ℤ) : ℤ := p^2 + q^2 + p * q^2

/-- The main theorem stating that √D is always irrational for consecutive even integers -/
theorem sqrt_D_irrational (p q : ℤ) (h : ConsecutiveEvenIntegers p q) :
  Irrational (Real.sqrt (D p q : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_sqrt_D_irrational_l3757_375727


namespace NUMINAMATH_CALUDE_map_scale_l3757_375731

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 : ℝ) * (real_km / map_cm) = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l3757_375731


namespace NUMINAMATH_CALUDE_circle_through_three_points_l3757_375797

theorem circle_through_three_points :
  let A : ℝ × ℝ := (1, 12)
  let B : ℝ × ℝ := (7, 10)
  let C : ℝ × ℝ := (-9, 2)
  let circle_equation (x y : ℝ) := x^2 + y^2 - 2*x - 4*y - 95 = 0
  (circle_equation A.1 A.2) ∧ 
  (circle_equation B.1 B.2) ∧ 
  (circle_equation C.1 C.2) := by
sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l3757_375797


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3757_375770

/-- The standard equation of a hyperbola with foci on the x-axis, given a and b -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

/-- Theorem: Given a = 3 and b = 4, the standard equation of the hyperbola with foci on the x-axis is (x²/9) - (y²/16) = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  let a : ℝ := 3
  let b : ℝ := 4
  hyperbola_equation x y a b ↔ (x^2 / 9) - (y^2 / 16) = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3757_375770


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3757_375739

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3757_375739


namespace NUMINAMATH_CALUDE_problem_solution_l3757_375717

theorem problem_solution (x y a b c d : ℝ) 
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c = -d) :
  (x + y)^3 - (-a*b)^2 + 3*c + 3*d = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3757_375717


namespace NUMINAMATH_CALUDE_ellipse_properties_l3757_375790

/-- An ellipse with minor axis length 2√3 and foci at (-1,0) and (1,0) -/
structure Ellipse where
  minor_axis : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  minor_axis_eq : minor_axis = 2 * Real.sqrt 3
  foci_eq : focus1 = (-1, 0) ∧ focus2 = (1, 0)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line y = x + m intersects the ellipse at two distinct points -/
def intersects_at_two_points (e : Ellipse) (m : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂, x₁ ≠ x₂ ∧ 
    standard_equation e x₁ y₁ ∧ 
    standard_equation e x₂ y₂ ∧
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m

theorem ellipse_properties (e : Ellipse) :
  (∀ x y, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m, intersects_at_two_points e m ↔ -Real.sqrt 7 < m ∧ m < Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3757_375790


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l3757_375755

/-- Given a rectangle with dimensions 15 inches by 10 inches and area 150 square inches,
    containing a shaded rectangle with area 110 square inches,
    the perimeter of the non-shaded region is 26 inches. -/
theorem non_shaded_perimeter (large_width large_height : ℝ)
                              (large_area shaded_area : ℝ)
                              (non_shaded_width non_shaded_height : ℝ) :
  large_width = 15 →
  large_height = 10 →
  large_area = 150 →
  shaded_area = 110 →
  large_area = large_width * large_height →
  non_shaded_width * non_shaded_height = large_area - shaded_area →
  non_shaded_width ≤ large_width →
  non_shaded_height ≤ large_height →
  2 * (non_shaded_width + non_shaded_height) = 26 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l3757_375755


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3757_375700

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x - 1)^2 = 25 ↔ x = 7/2 ∨ x = -3/2 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  1/3 * (x + 2)^3 - 9 = 0 ↔ x = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3757_375700


namespace NUMINAMATH_CALUDE_binary_to_base5_conversion_l3757_375728

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the base-5 number
def base5_num : ℕ := 140

-- Theorem statement
theorem binary_to_base5_conversion :
  (binary_num : ℕ).digits 5 = base5_num.digits 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base5_conversion_l3757_375728


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l3757_375701

theorem choose_three_from_ten (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose n k = 120 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l3757_375701


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_from_focus_distance_l3757_375788

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The semi-focal length of a hyperbola -/
def semi_focal_length (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity_from_focus_distance (h : Hyperbola) 
  (h_dist : focus_to_asymptote_distance h = (Real.sqrt 5 / 3) * semi_focal_length h) : 
  eccentricity h = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_from_focus_distance_l3757_375788


namespace NUMINAMATH_CALUDE_classroom_ratio_problem_l3757_375720

theorem classroom_ratio_problem (num_girls : ℕ) (ratio : ℚ) (num_boys : ℕ) : 
  num_girls = 10 → ratio = 1/2 → num_girls = ratio * num_boys → num_boys = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_problem_l3757_375720


namespace NUMINAMATH_CALUDE_hike_distance_l3757_375772

/-- Represents a 5-day hike with given conditions -/
structure FiveDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  first_two_days : day1 + day2 = 28
  second_fourth_avg : (day2 + day4) / 2 = 15
  last_three_days : day3 + day4 + day5 = 42
  first_third_days : day1 + day3 = 30

/-- The total distance of the hike is 70 miles -/
theorem hike_distance (h : FiveDayHike) : h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hike_distance_l3757_375772


namespace NUMINAMATH_CALUDE_exactly_one_divisible_by_3_5_7_l3757_375725

theorem exactly_one_divisible_by_3_5_7 :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 200 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_divisible_by_3_5_7_l3757_375725


namespace NUMINAMATH_CALUDE_largest_common_term_l3757_375769

theorem largest_common_term (n m : ℕ) : 
  (∃ n m : ℕ, 187 = 3 + 8 * n ∧ 187 = 5 + 9 * m) ∧ 
  (∀ k : ℕ, k > 187 → k ≤ 200 → ¬(∃ p q : ℕ, k = 3 + 8 * p ∧ k = 5 + 9 * q)) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_l3757_375769


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l3757_375710

theorem president_vice_president_selection (n : ℕ) (h : n = 5) :
  (n * (n - 1) : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l3757_375710


namespace NUMINAMATH_CALUDE_next_repeated_year_correct_l3757_375791

/-- A year consists of a repeated two-digit number if it can be written as ABAB where A and B are digits -/
def is_repeated_two_digit (year : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ year = a * 1000 + b * 100 + a * 10 + b

/-- The next year after 2020 with a repeated two-digit number -/
def next_repeated_year : ℕ := 2121

theorem next_repeated_year_correct :
  (next_repeated_year > 2020) ∧ 
  (is_repeated_two_digit next_repeated_year) ∧
  (∀ y : ℕ, 2020 < y ∧ y < next_repeated_year → ¬(is_repeated_two_digit y)) ∧
  (next_repeated_year - 2020 = 101) :=
sorry

end NUMINAMATH_CALUDE_next_repeated_year_correct_l3757_375791


namespace NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l3757_375709

theorem max_sum_of_factors (h s : ℕ) : h * s = 24 → h + s ≤ 25 := by
  sorry

theorem max_sum_of_factors_achieved : ∃ h s : ℕ, h * s = 24 ∧ h + s = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l3757_375709


namespace NUMINAMATH_CALUDE_sequence_formula_l3757_375721

/-- Given a sequence {aₙ} with sum Sₙ = (3/2)aₙ - 3, prove aₙ = 3 * 2^n -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (3/2) * a n - 3) :
  ∀ n : ℕ, n ≥ 1 → a n = 3 * 2^n := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3757_375721


namespace NUMINAMATH_CALUDE_vector_sum_squared_l3757_375778

variable (a b c m : ℝ × ℝ)

/-- m is the midpoint of a and b -/
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The squared norm of a 2D vector -/
def norm_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem vector_sum_squared (a b : ℝ × ℝ) :
  is_midpoint m a b →
  m = (4, 5) →
  dot_product a b = 12 →
  dot_product c (a.1 + b.1, a.2 + b.2) = 0 →
  norm_squared a + norm_squared b = 140 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_squared_l3757_375778


namespace NUMINAMATH_CALUDE_blood_expiration_date_l3757_375724

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The number of days in a non-leap year -/
def days_per_year : ℕ := 365

/-- The number of days in January -/
def days_in_january : ℕ := 31

/-- The number of days in February (non-leap year) -/
def days_in_february : ℕ := 28

/-- Calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 11

theorem blood_expiration_date :
  let total_days : ℕ := blood_expiration_time / seconds_per_day
  let days_in_second_year : ℕ := total_days - days_per_year
  let days_after_january : ℕ := days_in_second_year - days_in_january
  days_after_january = days_in_february + 8 :=
by sorry

end NUMINAMATH_CALUDE_blood_expiration_date_l3757_375724
