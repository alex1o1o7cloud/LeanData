import Mathlib

namespace NUMINAMATH_CALUDE_max_value_xyz_l257_25792

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  x^3 * y^3 * z^2 ≤ 4782969/390625 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l257_25792


namespace NUMINAMATH_CALUDE_part_one_part_two_l257_25789

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part 1
theorem part_one : (Set.univ \ A 1) ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) : A a ⊆ B ↔ a < -4 ∨ (0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l257_25789


namespace NUMINAMATH_CALUDE_sandwiches_per_person_l257_25731

def mini_croissants_per_set : ℕ := 12
def cost_per_set : ℕ := 8
def committee_size : ℕ := 24
def total_spent : ℕ := 32

theorem sandwiches_per_person :
  (total_spent / cost_per_set) * mini_croissants_per_set / committee_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_per_person_l257_25731


namespace NUMINAMATH_CALUDE_simplify_expressions_l257_25755

theorem simplify_expressions (x y a b : ℝ) :
  ((-3 * x + y) + (4 * x - 3 * y) = x - 2 * y) ∧
  (2 * a - (3 * b - 5 * a - (2 * a - 7 * b)) = 9 * a - 10 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l257_25755


namespace NUMINAMATH_CALUDE_product_of_integers_l257_25759

theorem product_of_integers (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 12)
  (h2 : a + b = 144) :
  (a : ℚ) * (b : ℚ) = 248832 / 169 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l257_25759


namespace NUMINAMATH_CALUDE_square_even_implies_even_l257_25777

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l257_25777


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l257_25751

-- Problem 1
theorem problem_1 : Real.sqrt 12 + (-2024)^(0 : ℕ) - 4 * Real.sin (60 * π / 180) = 1 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (x + 2)^2 + x * (x - 4) = 2 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l257_25751


namespace NUMINAMATH_CALUDE_binomial_product_integer_l257_25778

theorem binomial_product_integer (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2 * m) * Nat.factorial (2 * n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_integer_l257_25778


namespace NUMINAMATH_CALUDE_distance_calculation_l257_25790

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 94

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 10

/-- Time difference between Maxwell's start and Brad's start, in hours -/
def time_difference : ℝ := 1

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * maxwell_time + 
    brad_speed * (maxwell_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l257_25790


namespace NUMINAMATH_CALUDE_three_cards_same_suit_count_l257_25739

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- The number of ways to select three cards in order from the same suit -/
def ways_to_select_three_same_suit (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1) * (d.cards_per_suit - 2))

/-- Theorem stating the number of ways to select three cards from the same suit -/
theorem three_cards_same_suit_count (d : Deck) 
  (h2 : d.total_cards = 52) 
  (h3 : d.num_suits = 4) 
  (h4 : d.cards_per_suit = 13) : 
  ways_to_select_three_same_suit d = 6864 := by
  sorry

#eval ways_to_select_three_same_suit ⟨52, 4, 13, rfl⟩

end NUMINAMATH_CALUDE_three_cards_same_suit_count_l257_25739


namespace NUMINAMATH_CALUDE_greatest_length_segment_l257_25796

theorem greatest_length_segment (AE CD CF AC FD CE : ℝ) : 
  AE = Real.sqrt 106 →
  CD = 5 →
  CF = Real.sqrt 20 →
  AC = 5 →
  FD = Real.sqrt 85 →
  CE = Real.sqrt 29 →
  AC + CE > AE ∧ AC + CE > CD + CF ∧ AC + CE > AC + CF ∧ AC + CE > FD :=
by sorry

end NUMINAMATH_CALUDE_greatest_length_segment_l257_25796


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l257_25726

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) :
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l257_25726


namespace NUMINAMATH_CALUDE_shooting_scores_l257_25711

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

def variance_A : ℝ := 2.25
def variance_B : ℝ := 4.41

theorem shooting_scores :
  let avg_A := (scores_A.sum) / scores_A.length
  let avg_B := (scores_B.sum) / scores_B.length
  let avg_all := ((scores_A ++ scores_B).sum) / (scores_A.length + scores_B.length)
  avg_A < avg_B ∧ avg_all = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_shooting_scores_l257_25711


namespace NUMINAMATH_CALUDE_dress_price_calculation_l257_25793

-- Define the original price
def original_price : ℝ := 120

-- Define the discount rate
def discount_rate : ℝ := 0.30

-- Define the tax rate
def tax_rate : ℝ := 0.15

-- Define the total selling price
def total_selling_price : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem dress_price_calculation :
  total_selling_price = 96.6 := by sorry

end NUMINAMATH_CALUDE_dress_price_calculation_l257_25793


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l257_25705

theorem power_of_two_divisibility (n : ℕ) (hn : n ≥ 1) :
  (∃ k : ℕ, 2^n - 1 = 3 * k) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∃ l : ℕ, (2^n - 1) / 3 * l = 4 * m^2 + 1) →
  ∃ r : ℕ, n = 2^r :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l257_25705


namespace NUMINAMATH_CALUDE_women_in_first_group_l257_25795

/-- The number of women in the first group -/
def first_group : ℕ := 4

/-- The length of cloth colored by the first group in 2 days -/
def cloth_length_first_group : ℕ := 48

/-- The number of days taken by the first group to color the cloth -/
def days_first_group : ℕ := 2

/-- The number of women in the second group -/
def second_group : ℕ := 6

/-- The length of cloth colored by the second group in 1 day -/
def cloth_length_second_group : ℕ := 36

/-- The number of days taken by the second group to color the cloth -/
def days_second_group : ℕ := 1

theorem women_in_first_group : 
  first_group * cloth_length_second_group * days_first_group = 
  second_group * cloth_length_first_group * days_second_group :=
by sorry

end NUMINAMATH_CALUDE_women_in_first_group_l257_25795


namespace NUMINAMATH_CALUDE_f_increasing_m_range_l257_25735

/-- A function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x * |x - m| + 2 * x - 3

/-- Theorem stating that if f is increasing on ℝ, then m is in the interval [-2, 2] -/
theorem f_increasing_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_m_range_l257_25735


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l257_25775

theorem rachel_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 7)
  (h2 : math_homework = reading_homework + 3) :
  reading_homework = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l257_25775


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l257_25773

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ x ≤ 23 ∧ (1055 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1055 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l257_25773


namespace NUMINAMATH_CALUDE_reciprocal_sum_problem_l257_25748

theorem reciprocal_sum_problem (x y z : ℝ) 
  (h1 : 1/x + 1/y + 1/z = 2) 
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) : 
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_problem_l257_25748


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_four_is_smallest_k_l257_25733

/-- The quadratic equation 3x(kx-5)-x^2+7=0 has no real roots when k ≥ 4 -/
theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) ↔ k ≥ 4 :=
by sorry

/-- 4 is the smallest integer k for which 3x(kx-5)-x^2+7=0 has no real roots -/
theorem four_is_smallest_k : 
  ∀ k : ℤ, k < 4 → ∃ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_four_is_smallest_k_l257_25733


namespace NUMINAMATH_CALUDE_problem_solution_l257_25736

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l257_25736


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l257_25721

/-- The area of a right triangle with base 15 and height 10 is 75 -/
theorem right_triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 15 ∧ height = 10 ∧ area = (base * height) / 2 → area = 75

theorem right_triangle_area_proof : right_triangle_area 15 10 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l257_25721


namespace NUMINAMATH_CALUDE_power_of_64_two_thirds_l257_25769

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_two_thirds_l257_25769


namespace NUMINAMATH_CALUDE_inequality_condition_l257_25785

theorem inequality_condition (x y m : ℝ) : 
  x > 0 → 
  y > 0 → 
  2/x + 1/y = 1 → 
  (∀ x y, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1 → 2*x + y > m^2 + 8*m) ↔ 
  -9 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_condition_l257_25785


namespace NUMINAMATH_CALUDE_trey_kyle_turtle_difference_l257_25741

/-- Proves that Trey has 60 more turtles than Kyle given the conditions in the problem -/
theorem trey_kyle_turtle_difference : 
  ∀ (kristen trey kris layla tim kyle : ℚ),
  kristen = 24.5 →
  kris = kristen / 3 →
  trey = 8.5 * kris →
  layla = 2 * trey →
  tim = 2 / 3 * kristen →
  kyle = tim / 2 →
  trey - kyle = 60 := by
  sorry

end NUMINAMATH_CALUDE_trey_kyle_turtle_difference_l257_25741


namespace NUMINAMATH_CALUDE_probability_of_a_l257_25719

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.28571428571428575) :
  p a = 0.7142857142857143 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_a_l257_25719


namespace NUMINAMATH_CALUDE_circle_condition_l257_25703

/-- 
Theorem: The equation x^2 + y^2 + x + 2my + m = 0 represents a circle if and only if m ≠ 1/2.
-/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + x + 2*m*y + m = 0) ↔ m ≠ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l257_25703


namespace NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_l257_25770

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_deg_interior : ∃ (n : ℕ), n > 2 ∧ n * 150 = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_l257_25770


namespace NUMINAMATH_CALUDE_f_one_upper_bound_l257_25713

/-- A quadratic function f(x) = 2x^2 - mx + 5 where m is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -2],
    then f(1) ≤ 15 -/
theorem f_one_upper_bound (m : ℝ) 
  (h : ∀ x y, x ≤ y → y ≤ -2 → f m x ≥ f m y) : 
  f m 1 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_f_one_upper_bound_l257_25713


namespace NUMINAMATH_CALUDE_modulus_of_two_over_one_plus_i_l257_25784

open Complex

theorem modulus_of_two_over_one_plus_i :
  let z : ℂ := 2 / (1 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_two_over_one_plus_i_l257_25784


namespace NUMINAMATH_CALUDE_sum_of_roots_2016_l257_25743

/-- The function f(x) = x^2 - 2016x + 2015 -/
def f (x : ℝ) : ℝ := x^2 - 2016*x + 2015

/-- Theorem: If f(a) = f(b) = c for distinct a and b, then a + b = 2016 -/
theorem sum_of_roots_2016 (a b c : ℝ) (ha : f a = c) (hb : f b = c) (hab : a ≠ b) : a + b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_2016_l257_25743


namespace NUMINAMATH_CALUDE_sin_210_degrees_l257_25716

theorem sin_210_degrees :
  Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l257_25716


namespace NUMINAMATH_CALUDE_min_value_of_function_l257_25730

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = (a^(2/3) + b^(2/3))^(3/2) ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) →
    a / Real.sin θ + b / Real.cos θ ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l257_25730


namespace NUMINAMATH_CALUDE_polynomial_expansion_l257_25767

/-- Proves the expansion of (3z^3 + 4z^2 - 2z + 1)(2z^2 - 3z + 5) -/
theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 2 * z + 1) * (2 * z^2 - 3 * z + 5) =
  10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l257_25767


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l257_25757

/-- Fruit shop problem -/
theorem fruit_shop_problem 
  (may_total : ℝ) 
  (may_cost_A may_cost_B : ℝ)
  (june_cost_A june_cost_B : ℝ)
  (june_increase : ℝ)
  (june_total_quantity : ℝ)
  (h_may_total : may_total = 1700)
  (h_may_cost_A : may_cost_A = 8)
  (h_may_cost_B : may_cost_B = 18)
  (h_june_cost_A : june_cost_A = 10)
  (h_june_cost_B : june_cost_B = 20)
  (h_june_increase : june_increase = 300)
  (h_june_total_quantity : june_total_quantity = 120) :
  ∃ (may_quantity_A may_quantity_B : ℝ),
    may_quantity_A * may_cost_A + may_quantity_B * may_cost_B = may_total ∧
    may_quantity_A * june_cost_A + may_quantity_B * june_cost_B = may_total + june_increase ∧
    may_quantity_A = 100 ∧
    may_quantity_B = 50 ∧
    (∃ (june_quantity_A : ℝ),
      june_quantity_A ≤ 3 * (june_total_quantity - june_quantity_A) ∧
      june_quantity_A * june_cost_A + (june_total_quantity - june_quantity_A) * june_cost_B = 1500 ∧
      ∀ (other_june_quantity_A : ℝ),
        other_june_quantity_A ≤ 3 * (june_total_quantity - other_june_quantity_A) →
        other_june_quantity_A * june_cost_A + (june_total_quantity - other_june_quantity_A) * june_cost_B ≥ 1500) :=
by sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l257_25757


namespace NUMINAMATH_CALUDE_cost_price_is_65_l257_25787

/-- Given a cloth sale scenario, calculate the cost price per metre. -/
def cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price per metre is 65 given the problem conditions. -/
theorem cost_price_is_65 :
  cost_price_per_metre 300 18000 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_65_l257_25787


namespace NUMINAMATH_CALUDE_high_school_students_l257_25724

theorem high_school_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : music = 50)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  music + art - both + neither = 500 := by
  sorry

end NUMINAMATH_CALUDE_high_school_students_l257_25724


namespace NUMINAMATH_CALUDE_single_elimination_tournament_matches_l257_25776

/-- Calculates the number of matches in a single-elimination tournament. -/
def matches_played (num_players : ℕ) : ℕ :=
  num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players,
    511 matches are played to declare the winner. -/
theorem single_elimination_tournament_matches :
  matches_played 512 = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_matches_l257_25776


namespace NUMINAMATH_CALUDE_total_profit_is_56700_l257_25746

/-- Given a profit sharing ratio and c's profit, calculate the total profit -/
def calculate_total_profit (ratio_a ratio_b ratio_c : ℕ) (profit_c : ℕ) : ℕ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := profit_c / ratio_c
  total_parts * part_value

/-- Theorem: The total profit is $56,700 given the specified conditions -/
theorem total_profit_is_56700 :
  calculate_total_profit 8 9 10 21000 = 56700 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_56700_l257_25746


namespace NUMINAMATH_CALUDE_square_value_l257_25702

theorem square_value (a b : ℝ) (h : ∃ square, square * (3 * a * b) = 3 * a^2 * b) : 
  ∃ square, square = a := by sorry

end NUMINAMATH_CALUDE_square_value_l257_25702


namespace NUMINAMATH_CALUDE_present_age_of_b_l257_25749

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →
  (a = b + 7) →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_b_l257_25749


namespace NUMINAMATH_CALUDE_aqua_park_earnings_l257_25709

/-- Calculate the total earnings of an aqua park given admission cost, tour cost, and group sizes. -/
theorem aqua_park_earnings
  (admission_cost : ℕ)
  (tour_cost : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h1 : admission_cost = 12)
  (h2 : tour_cost = 6)
  (h3 : group1_size = 10)
  (h4 : group2_size = 5) :
  (group1_size * (admission_cost + tour_cost)) + (group2_size * admission_cost) = 240 := by
  sorry

#check aqua_park_earnings

end NUMINAMATH_CALUDE_aqua_park_earnings_l257_25709


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l257_25756

/-- Calculates the percentage of weight loss during beef processing. -/
theorem beef_weight_loss_percentage 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 876.9230769230769) 
  (h2 : weight_after = 570) : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l257_25756


namespace NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l257_25729

theorem cone_radius_from_slant_height_and_surface_area :
  ∀ (slant_height curved_surface_area : ℝ),
    slant_height = 22 →
    curved_surface_area = 483.80526865282815 →
    curved_surface_area = Real.pi * (7 : ℝ) * slant_height :=
by
  sorry

end NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l257_25729


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l257_25715

/-- Calculates the number of items drawn from a category in stratified sampling -/
def items_drawn (category_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (category_size * sample_size) / total_size

/-- Represents the stratified sampling problem -/
theorem stratified_sampling_sum (grains : ℕ) (vegetable_oil : ℕ) (animal_products : ℕ) (fruits_vegetables : ℕ) 
  (sample_size : ℕ) (h1 : grains = 40) (h2 : vegetable_oil = 10) (h3 : animal_products = 30) 
  (h4 : fruits_vegetables = 20) (h5 : sample_size = 20) :
  items_drawn vegetable_oil (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size + 
  items_drawn fruits_vegetables (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_sum_l257_25715


namespace NUMINAMATH_CALUDE_calculation_proof_l257_25768

theorem calculation_proof : (1/2)⁻¹ - 3 * Real.tan (30 * π / 180) + (1 - Real.pi)^0 + Real.sqrt 12 = Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l257_25768


namespace NUMINAMATH_CALUDE_simplification_condition_l257_25712

theorem simplification_condition (x y k : ℝ) : 
  y = k * x →
  ((x - y) * (2 * x - y) - 3 * x * (2 * x - y) = 5 * x^2) ↔ (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_simplification_condition_l257_25712


namespace NUMINAMATH_CALUDE_common_tangent_parabola_log_l257_25718

theorem common_tangent_parabola_log (a : ℝ) : 
  (∃ x₁ x₂ y : ℝ, 
    y = a * x₁^2 ∧ 
    y = Real.log x₂ ∧ 
    2 * a * x₁ = 2 ∧ 
    1 / x₂ = 2) → 
  a = 1 / Real.log (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_parabola_log_l257_25718


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l257_25710

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a_3 = 4 and a_7 = 12, prove that a_11 = 36 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_3 : a 3 = 4) 
    (h_7 : a 7 = 12) : 
  a 11 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l257_25710


namespace NUMINAMATH_CALUDE_product_equals_888888_l257_25753

theorem product_equals_888888 : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_888888_l257_25753


namespace NUMINAMATH_CALUDE_remainder_after_adding_2023_l257_25771

theorem remainder_after_adding_2023 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2023_l257_25771


namespace NUMINAMATH_CALUDE_max_volume_angle_l257_25717

/-- A square ABCD folded along diagonal AC to form a regular pyramid -/
structure FoldedSquare where
  side : ℝ
  fold_angle : ℝ

/-- The angle between line BD and plane ABC in the folded square -/
def angle_bd_abc (s : FoldedSquare) : ℝ := sorry

/-- The volume of the pyramid formed by the folded square -/
def pyramid_volume (s : FoldedSquare) : ℝ := sorry

theorem max_volume_angle (s : FoldedSquare) :
  (∀ t : FoldedSquare, pyramid_volume t ≤ pyramid_volume s) →
  angle_bd_abc s = 45 := by sorry

end NUMINAMATH_CALUDE_max_volume_angle_l257_25717


namespace NUMINAMATH_CALUDE_bridge_concrete_total_l257_25763

/-- The amount of concrete needed for a bridge -/
structure BridgeConcrete where
  roadway_deck : ℕ
  single_anchor : ℕ
  num_anchors : ℕ
  supporting_pillars : ℕ

/-- The total amount of concrete needed for the bridge -/
def total_concrete (b : BridgeConcrete) : ℕ :=
  b.roadway_deck + b.single_anchor * b.num_anchors + b.supporting_pillars

/-- Theorem: The total amount of concrete needed for the bridge is 4800 tons -/
theorem bridge_concrete_total :
  let b : BridgeConcrete := {
    roadway_deck := 1600,
    single_anchor := 700,
    num_anchors := 2,
    supporting_pillars := 1800
  }
  total_concrete b = 4800 := by sorry

end NUMINAMATH_CALUDE_bridge_concrete_total_l257_25763


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l257_25772

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℕ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l257_25772


namespace NUMINAMATH_CALUDE_average_distance_scientific_notation_l257_25798

-- Define the average distance between the Earth and the Sun
def average_distance : ℝ := 149600000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.496 * (10 ^ 8)

-- Theorem to prove the equivalence
theorem average_distance_scientific_notation : average_distance = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_average_distance_scientific_notation_l257_25798


namespace NUMINAMATH_CALUDE_min_value_abc_l257_25761

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 ≥ 18 ∧
  (a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 = 18 ↔ a = 3 ∧ b = 1/3 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l257_25761


namespace NUMINAMATH_CALUDE_constant_function_from_surjective_injective_l257_25714

theorem constant_function_from_surjective_injective
  (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_function_from_surjective_injective_l257_25714


namespace NUMINAMATH_CALUDE_interest_equality_second_sum_l257_25738

/-- Given a total sum divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, prove that the second part is equal to 1680 rupees. -/
theorem interest_equality_second_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) :
  total = 2730 →
  total = first_part + second_part →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1680 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_second_sum_l257_25738


namespace NUMINAMATH_CALUDE_handshakes_theorem_l257_25732

/-- Calculate the number of handshakes in a single meeting -/
def handshakes_in_meeting (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes in two meetings -/
def total_handshakes (first_meeting_attendees second_meeting_attendees overlap : ℕ) : ℕ :=
  handshakes_in_meeting first_meeting_attendees +
  handshakes_in_meeting second_meeting_attendees -
  handshakes_in_meeting overlap

/-- Prove that the total number of handshakes in the two meetings is 41 -/
theorem handshakes_theorem :
  let first_meeting_attendees : ℕ := 7
  let second_meeting_attendees : ℕ := 7
  let overlap : ℕ := 2
  total_handshakes first_meeting_attendees second_meeting_attendees overlap = 41 := by
  sorry

#eval total_handshakes 7 7 2

end NUMINAMATH_CALUDE_handshakes_theorem_l257_25732


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l257_25701

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The original statement (given as true)
axiom original_statement : ∀ (triangle : Type), is_equilateral triangle → is_isosceles triangle

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (triangle : Type), is_isosceles triangle ∧ ¬is_equilateral triangle) ∧
  (∃ (triangle : Type), ¬is_equilateral triangle ∧ is_isosceles triangle) :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l257_25701


namespace NUMINAMATH_CALUDE_choose_five_three_l257_25782

theorem choose_five_three (n : ℕ) (k : ℕ) : n = 5 ∧ k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_five_three_l257_25782


namespace NUMINAMATH_CALUDE_exchange_divisibility_l257_25754

theorem exchange_divisibility (p a d : ℤ) : 
  p = 4*a + d ∧ p = a + 5*d → 
  ∃ (t : ℤ), p = 19*t ∧ a = 4*t ∧ d = 3*t ∧ p + a + d = 26*t :=
by sorry

end NUMINAMATH_CALUDE_exchange_divisibility_l257_25754


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l257_25786

-- Define the function f(x) = x³ - 3x² - 9x
def f (x : ℝ) := x^3 - 3*x^2 - 9*x

-- Define the open interval (-2, 2)
def I := Set.Ioo (-2 : ℝ) 2

-- State the theorem
theorem extreme_values_of_f :
  (∃ (x : ℝ), x ∈ I ∧ f x = 5) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 5) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = -2) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l257_25786


namespace NUMINAMATH_CALUDE_first_day_cost_l257_25764

/-- The cost of a hamburger -/
def hamburger_cost : ℚ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℚ := 1

/-- The cost of 2 hamburgers and 3 hot dogs -/
def second_day_cost : ℚ := 7

theorem first_day_cost : 3 * hamburger_cost + 4 * hot_dog_cost = 10 :=
  by sorry

end NUMINAMATH_CALUDE_first_day_cost_l257_25764


namespace NUMINAMATH_CALUDE_smallest_NPP_l257_25727

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def last_two_digits_equal (n : ℕ) : Prop :=
  (n % 100) % 11 = 0

theorem smallest_NPP :
  ∃ (M N P : ℕ),
    is_two_digit_with_equal_digits (11 * M) ∧
    is_one_digit N ∧
    is_three_digit (100 * N + 10 * P + P) ∧
    11 * M * N = 100 * N + 10 * P + P ∧
    (∀ (M' N' P' : ℕ),
      is_two_digit_with_equal_digits (11 * M') →
      is_one_digit N' →
      is_three_digit (100 * N' + 10 * P' + P') →
      11 * M' * N' = 100 * N' + 10 * P' + P' →
      100 * N + 10 * P + P ≤ 100 * N' + 10 * P' + P') ∧
    M = 2 ∧ N = 3 ∧ P = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_NPP_l257_25727


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l257_25797

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3)
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (β - 2*α) = 4/3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l257_25797


namespace NUMINAMATH_CALUDE_square_root_equality_l257_25720

theorem square_root_equality (a b : ℝ) : 
  Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b) → a = 6 ∧ b = 35 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l257_25720


namespace NUMINAMATH_CALUDE_angle_equality_l257_25766

theorem angle_equality (θ : Real) (h1 : Real.sqrt 5 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) 
  (h2 : 0 < θ ∧ θ < π / 2) : θ = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l257_25766


namespace NUMINAMATH_CALUDE_owls_joined_l257_25791

def initial_owls : ℕ := 3
def final_owls : ℕ := 5

theorem owls_joined : final_owls - initial_owls = 2 := by
  sorry

end NUMINAMATH_CALUDE_owls_joined_l257_25791


namespace NUMINAMATH_CALUDE_third_circle_properties_l257_25745

/-- Given two concentric circles with radii 10 and 20 units, prove that a third circle
    with area equal to the shaded area between the two concentric circles has a radius
    of 10√3 and a circumference of 20√3π. -/
theorem third_circle_properties (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 20)
    (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) :
  r₃ = 10 * Real.sqrt 3 ∧ 2 * π * r₃ = 20 * Real.sqrt 3 * π := by
  sorry

#check third_circle_properties

end NUMINAMATH_CALUDE_third_circle_properties_l257_25745


namespace NUMINAMATH_CALUDE_shaded_area_square_minus_semicircles_l257_25794

/-- The area of a square with side length 14 cm minus the area of two semicircles 
    with diameters equal to the side length of the square is equal to 196 - 49π cm². -/
theorem shaded_area_square_minus_semicircles : 
  let side_length : ℝ := 14
  let square_area : ℝ := side_length ^ 2
  let semicircle_radius : ℝ := side_length / 2
  let semicircles_area : ℝ := π * semicircle_radius ^ 2
  square_area - semicircles_area = 196 - 49 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_minus_semicircles_l257_25794


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l257_25737

/-- Calculates the total profit for a partnership given investments and one partner's profit share -/
def calculate_total_profit (tom_investment : ℕ) (jose_investment : ℕ) (tom_months : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_total := tom_investment * tom_months
  let jose_total := jose_investment * jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * jose_profit) / (jose_total / (tom_total.gcd jose_total))

theorem partnership_profit_theorem (tom_investment jose_investment tom_months jose_months jose_profit : ℕ) 
  (h1 : tom_investment = 30000)
  (h2 : jose_investment = 45000)
  (h3 : tom_months = 12)
  (h4 : jose_months = 10)
  (h5 : jose_profit = 40000) :
  calculate_total_profit tom_investment jose_investment tom_months jose_months jose_profit = 72000 := by
  sorry

#eval calculate_total_profit 30000 45000 12 10 40000

end NUMINAMATH_CALUDE_partnership_profit_theorem_l257_25737


namespace NUMINAMATH_CALUDE_order_relationship_l257_25750

theorem order_relationship (a b c d : ℝ)
  (h1 : a < b)
  (h2 : c < d)
  (h3 : a + b < c + d)
  (h4 : a * b = c * d)
  (h5 : c * d < 0) :
  a < c ∧ c < b ∧ b < d := by
  sorry

end NUMINAMATH_CALUDE_order_relationship_l257_25750


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l257_25723

theorem min_value_sum_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 2) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l257_25723


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l257_25706

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < 3 → a * x^2 + x + b > 0) ∧
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) → a * x^2 + x + b ≤ 0) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l257_25706


namespace NUMINAMATH_CALUDE_third_pair_weight_l257_25747

def dumbbell_system (weight1 weight2 weight3 : ℕ) : Prop :=
  weight1 * 2 + weight2 * 2 + weight3 * 2 = 32

theorem third_pair_weight :
  ∃ (weight3 : ℕ), dumbbell_system 3 5 weight3 ∧ weight3 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_third_pair_weight_l257_25747


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l257_25708

theorem geometric_sequence_second_term 
  (a : ℕ → ℚ) -- a is the sequence
  (h1 : a 3 = 12) -- third term is 12
  (h2 : a 4 = 18) -- fourth term is 18
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 4 / a 3)) -- definition of geometric sequence
  : a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l257_25708


namespace NUMINAMATH_CALUDE_computer_software_price_sum_l257_25799

theorem computer_software_price_sum : 
  ∀ (b a : ℝ),
  (b + 0.3 * b = 351) →
  (a + 0.05 * a = 420) →
  2 * b + 2 * a = 1340 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_software_price_sum_l257_25799


namespace NUMINAMATH_CALUDE_complementary_events_l257_25742

-- Define the sample space for two shots
inductive ShotOutcome
| HH  -- Hit-Hit
| HM  -- Hit-Miss
| MH  -- Miss-Hit
| MM  -- Miss-Miss

-- Define the event of missing both times
def missBoth : Set ShotOutcome := {ShotOutcome.MM}

-- Define the event of hitting at least once
def hitAtLeastOnce : Set ShotOutcome := {ShotOutcome.HH, ShotOutcome.HM, ShotOutcome.MH}

-- Theorem stating that hitAtLeastOnce is the complement of missBoth
theorem complementary_events :
  hitAtLeastOnce = missBoth.compl :=
sorry

end NUMINAMATH_CALUDE_complementary_events_l257_25742


namespace NUMINAMATH_CALUDE_solution_value_l257_25707

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = -2 ∧ a * x - 6 = a + 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l257_25707


namespace NUMINAMATH_CALUDE_odd_integers_count_odd_integers_three_different_digits_count_l257_25762

theorem odd_integers_count : ℕ := by
  -- Define the range of integers
  let lower_bound : ℕ := 2000
  let upper_bound : ℕ := 3000

  -- Define the set of possible odd units digits
  let odd_units : Finset ℕ := {1, 3, 5, 7, 9}

  -- Define the count of choices for each digit position
  let thousands_choices : ℕ := 1  -- Always 2
  let hundreds_choices : ℕ := 8   -- Excluding 2 and the chosen units digit
  let tens_choices : ℕ := 7       -- Excluding 2, hundreds digit, and units digit
  let units_choices : ℕ := Finset.card odd_units

  -- Calculate the total count
  let total_count : ℕ := thousands_choices * hundreds_choices * tens_choices * units_choices

  -- Prove that the count equals 280
  sorry

-- The theorem statement
theorem odd_integers_three_different_digits_count :
  (odd_integers_count : ℕ) = 280 := by sorry

end NUMINAMATH_CALUDE_odd_integers_count_odd_integers_three_different_digits_count_l257_25762


namespace NUMINAMATH_CALUDE_mile_equals_400_rods_l257_25760

/-- Conversion rate from miles to furlongs -/
def mile_to_furlong : ℚ := 8

/-- Conversion rate from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

theorem mile_equals_400_rods : rods_in_mile = 400 := by
  sorry

end NUMINAMATH_CALUDE_mile_equals_400_rods_l257_25760


namespace NUMINAMATH_CALUDE_bus_trip_count_l257_25700

/-- Calculates the total number of people on a bus given the number of boys and additional information -/
def total_people_on_bus (num_boys : ℕ) : ℕ :=
  let num_girls : ℕ := num_boys + (2 * num_boys / 5)
  let num_students : ℕ := num_boys + num_girls
  let num_adults : ℕ := 3  -- driver, assistant, and teacher
  num_students + num_adults

/-- Proves that the total number of people on the bus is 123 given the problem conditions -/
theorem bus_trip_count : total_people_on_bus 50 = 123 := by
  sorry

#eval total_people_on_bus 50  -- This should output 123

end NUMINAMATH_CALUDE_bus_trip_count_l257_25700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l257_25704

/-- An arithmetic sequence {aₙ} where a₃ = 4 and a₅ = 8 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 4 ∧ a 5 = 8

/-- The 11th term of the arithmetic sequence is 20 -/
theorem arithmetic_sequence_11th_term (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) : a 11 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l257_25704


namespace NUMINAMATH_CALUDE_estimate_smaller_than_actual_l257_25734

theorem estimate_smaller_than_actual (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  (x - z) - (y + z) < x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_smaller_than_actual_l257_25734


namespace NUMINAMATH_CALUDE_odd_function_2019_l257_25779

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_2019 (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_sym : ∀ x, f (1 + x) = f (1 - x))
  (h_f1 : f 1 = 9) :
  f 2019 = -9 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_2019_l257_25779


namespace NUMINAMATH_CALUDE_smallest_c_value_l257_25725

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * ↑x - d) = Real.cos (35 * ↑x)) :
  c ≥ 35 ∧ ∀ c' ≥ 0, (∀ x : ℤ, Real.cos (c' * ↑x - d) = Real.cos (35 * ↑x)) → c' ≥ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l257_25725


namespace NUMINAMATH_CALUDE_joshua_skittles_l257_25744

/-- The number of friends Joshua has -/
def num_friends : ℕ := 5

/-- The number of Skittles each friend would get if Joshua shares them equally -/
def skittles_per_friend : ℕ := 8

/-- The total number of Skittles Joshua has -/
def total_skittles : ℕ := num_friends * skittles_per_friend

/-- Theorem: Joshua has 40 Skittles -/
theorem joshua_skittles : total_skittles = 40 := by
  sorry

end NUMINAMATH_CALUDE_joshua_skittles_l257_25744


namespace NUMINAMATH_CALUDE_dinners_sold_in_four_days_l257_25788

/-- Calculates the total number of dinners sold over 4 days given specific sales patterns. -/
def total_dinners_sold (monday : ℕ) : ℕ :=
  let tuesday := monday + 40
  let wednesday := tuesday / 2
  let thursday := wednesday + 3
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific sales pattern, 203 dinners were sold over 4 days. -/
theorem dinners_sold_in_four_days : total_dinners_sold 40 = 203 := by
  sorry

end NUMINAMATH_CALUDE_dinners_sold_in_four_days_l257_25788


namespace NUMINAMATH_CALUDE_constant_distance_l257_25781

/-- Ellipse E with eccentricity 1/2 and area of triangle F₁PF₂ equal to 3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a^2/4 + b^2/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Point on the line y = 2√3 -/
structure PointOnLine where
  x : ℝ
  y : ℝ
  h : y = 2 * Real.sqrt 3

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (M : PointOnEllipse E) (N : PointOnLine) 
  (h : (M.x * N.x + M.y * N.y) / (M.x^2 + M.y^2).sqrt / (N.x^2 + N.y^2).sqrt = 0) :
  ((M.y * N.x - M.x * N.y)^2 / ((M.x - N.x)^2 + (M.y - N.y)^2)).sqrt = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_distance_l257_25781


namespace NUMINAMATH_CALUDE_floor_ceil_calculation_l257_25780

theorem floor_ceil_calculation : 
  ⌊(18 : ℝ) / 5 * (-33 : ℝ) / 4⌋ - ⌈(18 : ℝ) / 5 * ⌈(-33 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_calculation_l257_25780


namespace NUMINAMATH_CALUDE_counterexample_exists_l257_25758

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l257_25758


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l257_25765

theorem min_value_of_quadratic_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l257_25765


namespace NUMINAMATH_CALUDE_geometric_sequence_linear_system_l257_25752

theorem geometric_sequence_linear_system (a : ℕ → ℝ) (q : ℝ) (h : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (∃ x y : ℝ, a 1 * x + a 3 * y = 2 ∧ a 2 * x + a 4 * y = 1) ↔ q = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_linear_system_l257_25752


namespace NUMINAMATH_CALUDE_correct_regression_coefficients_l257_25722

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define positive correlation
def positively_correlated (a : ℝ) : Prop := a > 0

-- Define the sample means
def x_mean : ℝ := 3
def y_mean : ℝ := 3.5

-- Theorem statement
theorem correct_regression_coefficients (a b : ℝ) :
  positively_correlated a ∧
  linear_regression a b x_mean = y_mean →
  a = 0.4 ∧ b = 2.3 :=
by sorry

end NUMINAMATH_CALUDE_correct_regression_coefficients_l257_25722


namespace NUMINAMATH_CALUDE_balance_theorem_l257_25774

/-- Represents the balance of symbols -/
structure Balance :=
  (star : ℚ)
  (square : ℚ)
  (heart : ℚ)
  (club : ℚ)

/-- The balance equations from the problem -/
def balance_equations (b : Balance) : Prop :=
  3 * b.star + 4 * b.square + b.heart = 12 * b.club ∧
  b.star = b.heart + 2 * b.club

/-- The theorem to prove -/
theorem balance_theorem (b : Balance) :
  balance_equations b →
  3 * b.square + 2 * b.heart = (26 / 9) * b.square :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l257_25774


namespace NUMINAMATH_CALUDE_star_seven_three_l257_25740

def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

theorem star_seven_three : star 7 3 = 16 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l257_25740


namespace NUMINAMATH_CALUDE_binomial_20_4_l257_25728

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l257_25728


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l257_25783

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 5) ↔ (x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l257_25783
