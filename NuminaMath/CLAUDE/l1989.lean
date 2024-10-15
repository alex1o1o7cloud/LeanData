import Mathlib

namespace NUMINAMATH_CALUDE_vase_transport_problem_l1989_198989

theorem vase_transport_problem (x : ℕ) : 
  (∃ C : ℝ, 
    (10 * (x - 50) - C = -300) ∧ 
    (12 * (x - 50) - C = 800)) → 
  x = 600 := by
  sorry

end NUMINAMATH_CALUDE_vase_transport_problem_l1989_198989


namespace NUMINAMATH_CALUDE_sum_of_digits_2008_5009_7_l1989_198956

theorem sum_of_digits_2008_5009_7 :
  ∃ (n : ℕ), n = 2^2008 * 5^2009 * 7 ∧ (List.sum (Nat.digits 10 n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2008_5009_7_l1989_198956


namespace NUMINAMATH_CALUDE_quinary_324_equals_binary_1011001_l1989_198972

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun ⟨i, d⟩ acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem quinary_324_equals_binary_1011001 :
  decimal_to_binary (quinary_to_decimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_quinary_324_equals_binary_1011001_l1989_198972


namespace NUMINAMATH_CALUDE_sandy_fish_count_l1989_198940

/-- The number of pet fish Sandy has after buying more -/
def total_fish (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sandy now has 32 pet fish -/
theorem sandy_fish_count : total_fish 26 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l1989_198940


namespace NUMINAMATH_CALUDE_square_of_sum_l1989_198971

theorem square_of_sum (x y : ℝ) 
  (h1 : x * (2 * x + y) = 36)
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_square_of_sum_l1989_198971


namespace NUMINAMATH_CALUDE_xiaoxiao_types_faster_l1989_198942

/-- Represents a typist with their typing data -/
structure Typist where
  name : String
  characters : ℕ
  minutes : ℕ

/-- Calculate the typing speed in characters per minute -/
def typingSpeed (t : Typist) : ℚ :=
  t.characters / t.minutes

/-- Determine if one typist is faster than another -/
def isFaster (t1 t2 : Typist) : Prop :=
  typingSpeed t1 > typingSpeed t2

theorem xiaoxiao_types_faster :
  let taoqi : Typist := { name := "淘气", characters := 200, minutes := 5 }
  let xiaoxiao : Typist := { name := "笑笑", characters := 132, minutes := 3 }
  isFaster xiaoxiao taoqi := by
  sorry

end NUMINAMATH_CALUDE_xiaoxiao_types_faster_l1989_198942


namespace NUMINAMATH_CALUDE_solve_star_equation_l1989_198952

-- Define the * operation
def star_op (a b : ℚ) : ℚ :=
  if a ≥ b then a^2 * b else a * b^2

-- Theorem statement
theorem solve_star_equation :
  ∃! m : ℚ, star_op 3 m = 48 ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1989_198952


namespace NUMINAMATH_CALUDE_largest_x_and_ratio_l1989_198949

theorem largest_x_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (5 * x / 7 + 1 = 3 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-7 + 21 * Real.sqrt 22) / 10) ∧
  (a * c * d / b = -70) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_ratio_l1989_198949


namespace NUMINAMATH_CALUDE_remainder_51_pow_2015_mod_13_l1989_198907

theorem remainder_51_pow_2015_mod_13 : 51^2015 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_51_pow_2015_mod_13_l1989_198907


namespace NUMINAMATH_CALUDE_f_upper_bound_l1989_198905

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 3

theorem f_upper_bound :
  ∃ (a : ℝ), a = 3 ∧ (∀ x ∈ Set.Icc (-2) 2, f x ≤ a) ∧
  (∀ b < a, ∃ x ∈ Set.Icc (-2) 2, f x > b) :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_l1989_198905


namespace NUMINAMATH_CALUDE_solve_equation_l1989_198988

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1989_198988


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1989_198951

/-- Theorem: Area change of a rectangle after length decrease and width increase -/
theorem rectangle_area_change
  (l w : ℝ)  -- l: original length, w: original width
  (hl : l > 0)  -- length is positive
  (hw : w > 0)  -- width is positive
  : (0.9 * l) * (1.3 * w) = 1.17 * (l * w) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1989_198951


namespace NUMINAMATH_CALUDE_euler_line_equation_l1989_198953

/-- Triangle ABC with vertices A(1,3) and B(2,1), and |AC| = |BC| -/
structure Triangle :=
  (C : ℝ × ℝ)
  (ac_eq_bc : (C.1 - 1)^2 + (C.2 - 3)^2 = (C.2 - 2)^2 + (C.2 - 1)^2)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0}

/-- Theorem: The Euler line of triangle ABC is 2x - 4y + 5 = 0 -/
theorem euler_line_equation (t : Triangle) : 
  EulerLine t = {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0} := by
  sorry

end NUMINAMATH_CALUDE_euler_line_equation_l1989_198953


namespace NUMINAMATH_CALUDE_avery_donation_l1989_198915

/-- The number of shirts Avery puts in the donation box -/
def num_shirts : ℕ := 4

/-- The number of pants Avery puts in the donation box -/
def num_pants : ℕ := 2 * num_shirts

/-- The number of shorts Avery puts in the donation box -/
def num_shorts : ℕ := num_pants / 2

/-- The total number of clothes Avery is donating -/
def total_clothes : ℕ := num_shirts + num_pants + num_shorts

theorem avery_donation : total_clothes = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l1989_198915


namespace NUMINAMATH_CALUDE_jinho_money_problem_l1989_198954

theorem jinho_money_problem (initial_money : ℕ) : 
  (initial_money / 2 + 300 + 
   ((initial_money - (initial_money / 2 + 300)) / 2 + 400) = initial_money) → 
  initial_money = 2200 :=
by sorry

end NUMINAMATH_CALUDE_jinho_money_problem_l1989_198954


namespace NUMINAMATH_CALUDE_city_population_l1989_198981

/-- Proves that given the conditions about women and retail workers in a city,
    the total population is 6,000,000 -/
theorem city_population (total_population : ℕ) : 
  (total_population / 2 : ℚ) = (total_population : ℚ) * (1 / 2 : ℚ) →
  ((total_population / 2 : ℚ) * (1 / 3 : ℚ) : ℚ) = 1000000 →
  total_population = 6000000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l1989_198981


namespace NUMINAMATH_CALUDE_inequality_proof_l1989_198990

theorem inequality_proof (n : ℕ) (a b : ℝ) 
  (h1 : n ≠ 1) (h2 : a > b) (h3 : b > 0) : 
  ((a + b) / 2) ^ n < (a ^ n + b ^ n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1989_198990


namespace NUMINAMATH_CALUDE_new_average_age_l1989_198966

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → initial_avg = 14 → new_person_age = 34 → 
  ((n : ℝ) * initial_avg + new_person_age) / ((n : ℝ) + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1989_198966


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_area_l1989_198995

theorem right_triangle_hypotenuse_and_area 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 60) (h_b : b = 80) : 
  c = 100 ∧ (1/2 * a * b) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_area_l1989_198995


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_property_l1989_198975

theorem smallest_k_for_divisibility_property (n : ℕ) :
  let M := Finset.range n
  (∃ k : ℕ, k > 0 ∧
    (∀ S : Finset ℕ, S ⊆ M → S.card = k →
      ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
    (∀ k' : ℕ, k' < k →
      ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
        ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b))) →
  let k := ⌈(n : ℚ) / 2⌉₊ + 1
  (∀ S : Finset ℕ, S ⊆ M → S.card = k →
    ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_property_l1989_198975


namespace NUMINAMATH_CALUDE_opposite_of_negative_twelve_l1989_198944

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_negative_twelve : opposite (-12) = 12 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_twelve_l1989_198944


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1989_198961

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The sample space of drawing two balls from the bag -/
def SampleSpace (b : Bag) := Fin (b.red + b.black) × Fin (b.red + b.black - 1)

/-- Event of drawing exactly one black ball -/
def ExactlyOneBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Event of drawing exactly two black balls -/
def ExactlyTwoBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Two events are mutually exclusive -/
def MutuallyExclusive {α : Type*} (A B : Set α) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary -/
def Complementary {α : Type*} (A B : Set α) : Prop :=
  MutuallyExclusive A B ∧ A ∪ B = Set.univ

/-- The main theorem -/
theorem mutually_exclusive_but_not_complementary :
  let b : Bag := ⟨2, 2⟩
  MutuallyExclusive (ExactlyOneBlack b) (ExactlyTwoBlack b) ∧
  ¬Complementary (ExactlyOneBlack b) (ExactlyTwoBlack b) := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1989_198961


namespace NUMINAMATH_CALUDE_sequence_nonpositive_l1989_198945

theorem sequence_nonpositive (N : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i ∈ Finset.range (N - 1), a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_nonpositive_l1989_198945


namespace NUMINAMATH_CALUDE_min_value_polynomial_l1989_198920

theorem min_value_polynomial (x y : ℝ) : 
  x^2 + y^2 - 6*x + 8*y + 7 ≥ -18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l1989_198920


namespace NUMINAMATH_CALUDE_min_overlap_brown_eyes_and_lunch_box_l1989_198941

theorem min_overlap_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 40) 
  (h2 : brown_eyes = 18) 
  (h3 : lunch_box = 25) : 
  ∃ (overlap : ℕ), 
    overlap ≥ brown_eyes + lunch_box - total_students ∧ 
    overlap = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_brown_eyes_and_lunch_box_l1989_198941


namespace NUMINAMATH_CALUDE_geometric_sequence_max_first_term_l1989_198926

theorem geometric_sequence_max_first_term 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 ≥ 1)
  (h_a2 : a 2 ≤ 2)
  (h_a3 : a 3 ≥ 3) :
  a 1 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_first_term_l1989_198926


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1989_198923

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, (x + 4) * (x + 3) ≥ 0 → x^2 + y^2 + 4*x + 3 ≤ 0) ∧
  (∃ x y, x^2 + y^2 + 4*x + 3 ≤ 0 ∧ (x + 4) * (x + 3) < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1989_198923


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l1989_198903

theorem smallest_perfect_square_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (n % 4 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → (k % 4 = 0) → (k % 5 = 0) → k ≥ n) ∧
  n = 400 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l1989_198903


namespace NUMINAMATH_CALUDE_gumball_probability_l1989_198927

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  blue_prob + pink_prob = 1 →
  blue_prob * blue_prob = 16 / 36 →
  pink_prob = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_gumball_probability_l1989_198927


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1989_198932

theorem binomial_coefficient_problem (a : ℝ) : 
  (6 : ℕ).choose 1 * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1989_198932


namespace NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1989_198906

/-- The probability of the Chinese team winning the gold medal in Women's Singles Table Tennis -/
theorem chinese_team_gold_medal_probability :
  let prob_A : ℚ := 3/7  -- Probability of Player A winning
  let prob_B : ℚ := 1/4  -- Probability of Player B winning
  -- Assuming the events are mutually exclusive
  prob_A + prob_B = 19/28 := by sorry

end NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1989_198906


namespace NUMINAMATH_CALUDE_distributive_property_implication_l1989_198967

theorem distributive_property_implication (a b c : ℝ) (h : c ≠ 0) :
  (∀ x y z : ℝ, (x + y) * z = x * z + y * z) →
  (a + b) / c = a / c + b / c :=
by sorry

end NUMINAMATH_CALUDE_distributive_property_implication_l1989_198967


namespace NUMINAMATH_CALUDE_gcf_of_40_120_80_l1989_198977

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_80_l1989_198977


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1989_198938

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1989_198938


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1989_198939

/-- Calculate the simple interest rate given principal, final amount, and time -/
theorem simple_interest_rate (principal final_amount time : ℝ) 
  (h_principal : principal > 0)
  (h_final : final_amount > principal)
  (h_time : time > 0) :
  (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1989_198939


namespace NUMINAMATH_CALUDE_rock_mist_distance_l1989_198970

/-- The distance from the city to Sky Falls in miles -/
def distance_to_sky_falls : ℝ := 8

/-- The factor by which Rock Mist Mountains are farther from the city than Sky Falls -/
def rock_mist_factor : ℝ := 50

/-- The distance from the city to Rock Mist Mountains in miles -/
def distance_to_rock_mist : ℝ := distance_to_sky_falls * rock_mist_factor

theorem rock_mist_distance : distance_to_rock_mist = 400 := by
  sorry

end NUMINAMATH_CALUDE_rock_mist_distance_l1989_198970


namespace NUMINAMATH_CALUDE_expression_evaluation_l1989_198919

theorem expression_evaluation : 
  (3 - 4 * (5 - 6)⁻¹)⁻¹ * (1 - 2⁻¹) = (1 : ℚ) / 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1989_198919


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1989_198943

theorem quadratic_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = -10 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 + 10*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1989_198943


namespace NUMINAMATH_CALUDE_average_chapters_per_book_l1989_198969

theorem average_chapters_per_book (total_chapters : Real) (total_books : Real) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) :
  total_chapters / total_books = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_average_chapters_per_book_l1989_198969


namespace NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_1_mod_6_l1989_198924

theorem unique_prime_between_30_and_40_with_remainder_1_mod_6 :
  ∃! n : ℕ, 30 < n ∧ n < 40 ∧ Nat.Prime n ∧ n % 6 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_1_mod_6_l1989_198924


namespace NUMINAMATH_CALUDE_corrected_mean_is_36_4_l1989_198900

/-- Calculates the corrected mean of a set of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  let original_sum := n * original_mean
  let difference := correct_value - wrong_value
  let corrected_sum := original_sum + difference
  corrected_sum / n

/-- Proves that the corrected mean is 36.4 given the specified conditions --/
theorem corrected_mean_is_36_4 :
  corrected_mean 50 36 23 43 = 36.4 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_is_36_4_l1989_198900


namespace NUMINAMATH_CALUDE_X_is_element_of_Y_l1989_198974

def X : Set Nat := {0, 1}

def Y : Set (Set Nat) := {s | s ⊆ X}

theorem X_is_element_of_Y : X ∈ Y := by sorry

end NUMINAMATH_CALUDE_X_is_element_of_Y_l1989_198974


namespace NUMINAMATH_CALUDE_pants_and_belt_price_difference_l1989_198931

def price_difference (total_cost pants_price : ℝ) : ℝ :=
  total_cost - 2 * pants_price

theorem pants_and_belt_price_difference :
  let total_cost : ℝ := 70.93
  let pants_price : ℝ := 34
  pants_price < (total_cost - pants_price) →
  price_difference total_cost pants_price = 2.93 := by
sorry

end NUMINAMATH_CALUDE_pants_and_belt_price_difference_l1989_198931


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1989_198947

theorem reciprocal_problem :
  (∀ x : ℚ, x ≠ 0 → x * (1 / x) = 1) →
  (1 / 0.125 = 8) ∧ (1 / 1 = 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1989_198947


namespace NUMINAMATH_CALUDE_equation_solution_l1989_198912

theorem equation_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1989_198912


namespace NUMINAMATH_CALUDE_circle_center_l1989_198934

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 12 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle given by x^2 - 6x + y^2 + 2y - 12 = 0 is (3, -1) -/
theorem circle_center : 
  ∃ (h k : ℝ), CircleCenter h k = (3, -1) ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 12 + 6*h - 2*k) :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1989_198934


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l1989_198913

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c form a basis in space V, and real numbers x, y, z 
    satisfy the equation x*a + y*b + z*c = 0, then x^2 + y^2 + z^2 = 0 -/
theorem sum_of_squares_zero (a b c : V) (x y z : ℝ) 
  (h_basis : LinearIndependent ℝ ![a, b, c]) 
  (h_eq : x • a + y • b + z • c = 0) : 
  x^2 + y^2 + z^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l1989_198913


namespace NUMINAMATH_CALUDE_robin_hair_cut_l1989_198965

/-- Calculates the length of hair cut off given initial length, growth, and final length -/
def hair_cut_length (initial_length growth final_length : ℝ) : ℝ :=
  initial_length + growth - final_length

/-- Theorem stating that given the conditions in the problem, Robin cut off 11 inches of hair -/
theorem robin_hair_cut :
  let initial_length : ℝ := 16
  let growth : ℝ := 12
  let final_length : ℝ := 17
  hair_cut_length initial_length growth final_length = 11 := by
  sorry

end NUMINAMATH_CALUDE_robin_hair_cut_l1989_198965


namespace NUMINAMATH_CALUDE_triangle_ratio_bound_l1989_198950

/-- For any triangle with perimeter p, circumradius R, and inradius r,
    the expression p/R * (1 - r/(3R)) is at most 5√3/2 -/
theorem triangle_ratio_bound (p R r : ℝ) (hp : p > 0) (hR : R > 0) (hr : r > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p = a + b + c ∧
    R = (a * b * c) / (4 * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
    r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * p)) :
  p / R * (1 - r / (3 * R)) ≤ 5 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_bound_l1989_198950


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1989_198997

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 →
  b = 36 / 99 →
  a / b = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1989_198997


namespace NUMINAMATH_CALUDE_m_range_theorem_l1989_198994

-- Define the conditions
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 4 = 0 ∧ x₂^2 + m*x₂ + 4 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ (Set.Ioo 1 3) ∪ (Set.Ioi 4) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1989_198994


namespace NUMINAMATH_CALUDE_lukes_mother_ten_bills_l1989_198968

def school_fee : ℕ := 350

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2

def father_fifty : ℕ := 4
def father_twenty : ℕ := 1
def father_ten : ℕ := 1

theorem lukes_mother_ten_bills (mother_ten : ℕ) :
  mother_fifty * 50 + mother_twenty * 20 + mother_ten * 10 +
  father_fifty * 50 + father_twenty * 20 + father_ten * 10 = school_fee →
  mother_ten = 3 := by
  sorry

end NUMINAMATH_CALUDE_lukes_mother_ten_bills_l1989_198968


namespace NUMINAMATH_CALUDE_james_tshirt_purchase_l1989_198948

/-- The total cost for a discounted purchase of t-shirts -/
def discounted_total_cost (num_shirts : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_shirts * original_price * (1 - discount_percent)

/-- Theorem: James pays $60 for 6 t-shirts at 50% off, originally priced at $20 each -/
theorem james_tshirt_purchase : 
  discounted_total_cost 6 20 (1/2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_tshirt_purchase_l1989_198948


namespace NUMINAMATH_CALUDE_quadratic_roots_triangle_range_l1989_198998

/-- Given a quadratic equation x^2 - 2x + m = 0 with two real roots a and b,
    where a, b, and 1 can form the sides of a triangle, prove that 3/4 < m ≤ 1 --/
theorem quadratic_roots_triangle_range (m : ℝ) (a b : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ x = a ∨ x = b) → 
  (a + b > 1 ∧ a > 0 ∧ b > 0 ∧ 1 > 0 ∧ a + 1 > b ∧ b + 1 > a) →
  (3/4 < m ∧ m ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_triangle_range_l1989_198998


namespace NUMINAMATH_CALUDE_road_width_calculation_l1989_198960

/-- Calculates the width of roads on a rectangular lawn given the dimensions and cost --/
theorem road_width_calculation (length width total_cost cost_per_sqm : ℝ) : 
  length = 80 →
  width = 60 →
  total_cost = 3900 →
  cost_per_sqm = 3 →
  let road_area := total_cost / cost_per_sqm
  let road_width := road_area / (length + width)
  road_width = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_road_width_calculation_l1989_198960


namespace NUMINAMATH_CALUDE_divisibility_problem_l1989_198976

theorem divisibility_problem (N : ℕ) : 
  N % 44 = 0 → N % 39 = 15 → N / 44 = 3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1989_198976


namespace NUMINAMATH_CALUDE_no_solution_equation_l1989_198910

theorem no_solution_equation : 
  ¬∃ (x : ℝ), (2 / (x + 1) + 3 / (x - 1) = 6 / (x^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1989_198910


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1989_198964

theorem complex_fraction_evaluation : 
  (2 / (3 + 1/5) + ((3 + 1/4) / 13) / (2/3) + (2 + 5/18 - 17/36) * (18/65)) * (1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1989_198964


namespace NUMINAMATH_CALUDE_textbook_selling_price_l1989_198929

/-- The selling price of a textbook, given its cost price and profit -/
theorem textbook_selling_price (cost_price profit : ℝ) (h1 : cost_price = 44) (h2 : profit = 11) :
  cost_price + profit = 55 := by
  sorry

#check textbook_selling_price

end NUMINAMATH_CALUDE_textbook_selling_price_l1989_198929


namespace NUMINAMATH_CALUDE_only_log23_not_computable_l1989_198983

-- Define the given logarithm values
def log27 : ℝ := 1.4314
def log32 : ℝ := 1.5052

-- Define a function to represent the computability of a logarithm
def is_computable (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log27 log32 = Real.log x

-- State the theorem
theorem only_log23_not_computable :
  ¬(is_computable 23) ∧ 
  (is_computable (9/8)) ∧ 
  (is_computable 28) ∧ 
  (is_computable 800) ∧ 
  (is_computable 0.45) := by
  sorry

end NUMINAMATH_CALUDE_only_log23_not_computable_l1989_198983


namespace NUMINAMATH_CALUDE_three_sides_form_triangle_l1989_198935

/-- A polygon circumscribed around a circle -/
structure CircumscribedPolygon where
  n : ℕ
  sides : Fin n → ℝ
  is_circumscribed : Bool

/-- The theorem stating that in any polygon circumscribed around a circle 
    with at least 4 sides, there exist three sides that can form a triangle -/
theorem three_sides_form_triangle (P : CircumscribedPolygon) 
  (h : P.n ≥ 4) (h_circ : P.is_circumscribed = true) :
  ∃ (i j k : Fin P.n), 
    P.sides i + P.sides j > P.sides k ∧
    P.sides j + P.sides k > P.sides i ∧
    P.sides k + P.sides i > P.sides j :=
sorry


end NUMINAMATH_CALUDE_three_sides_form_triangle_l1989_198935


namespace NUMINAMATH_CALUDE_min_students_l1989_198982

/-- Represents the number of students in each income group -/
structure IncomeGroups where
  low : ℕ
  middle : ℕ
  high : ℕ

/-- Represents the lowest salary in each income range -/
structure LowestSalaries where
  low : ℝ
  middle : ℝ
  high : ℝ

/-- Represents the median salary in each income range -/
def medianSalaries (lowest : LowestSalaries) : LowestSalaries :=
  { low := lowest.low + 50000
  , middle := lowest.middle + 40000
  , high := lowest.high + 30000 }

/-- The conditions of the problem -/
structure GraduatingClass where
  groups : IncomeGroups
  lowest : LowestSalaries
  salary_range : ℝ
  average_salary : ℝ
  median_salary : ℝ
  (high_twice_low : groups.high = 2 * groups.low)
  (middle_sum_others : groups.middle = groups.low + groups.high)
  (salary_range_constant : ∀ (r : LowestSalaries → ℝ), r (medianSalaries lowest) - r lowest = salary_range)
  (average_above_median : average_salary = median_salary + 20000)

/-- The theorem to prove -/
theorem min_students (c : GraduatingClass) : 
  c.groups.low + c.groups.middle + c.groups.high ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_students_l1989_198982


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_a_greater_than_one_l1989_198992

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem 2: (∁ₗA) ∩ B = {x | 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 1
theorem a_greater_than_one (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_a_greater_than_one_l1989_198992


namespace NUMINAMATH_CALUDE_melissa_total_score_l1989_198911

/-- Given a player who scores the same number of points in each game,
    calculate their total score across multiple games. -/
def totalPoints (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Theorem: A player scoring 120 points per game for 10 games
    will have a total score of 1200 points. -/
theorem melissa_total_score :
  totalPoints 120 10 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_melissa_total_score_l1989_198911


namespace NUMINAMATH_CALUDE_tan_negative_4095_degrees_l1989_198902

theorem tan_negative_4095_degrees : Real.tan ((-4095 : ℝ) * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_4095_degrees_l1989_198902


namespace NUMINAMATH_CALUDE_infinitely_many_satisfy_property_l1989_198955

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property that n divides F_{F_n} but not F_n -/
def satisfies_property (n : ℕ) : Prop :=
  n > 0 ∧ (n ∣ fib (fib n)) ∧ ¬(n ∣ fib n)

theorem infinitely_many_satisfy_property :
  ∀ k : ℕ, k > 0 → satisfies_property (12 * k) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfy_property_l1989_198955


namespace NUMINAMATH_CALUDE_cube_root_equation_l1989_198946

theorem cube_root_equation (x : ℝ) : 
  (x * (x^3)^(1/2))^(1/3) = 3 → x = 3^(6/5) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l1989_198946


namespace NUMINAMATH_CALUDE_purchase_price_l1989_198922

/-- The purchase price of an article given markup conditions -/
theorem purchase_price (M : ℝ) (P : ℝ) : M = 0.30 * P + 12 → M = 55 → P = 143.33 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_l1989_198922


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1989_198909

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) : 
  x^4 + y^4 = 82 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1989_198909


namespace NUMINAMATH_CALUDE_unique_satisfying_pair_l1989_198978

/-- A pair of real numbers satisfying both arithmetic and geometric progression conditions -/
def SatisfyingPair (a b : ℝ) : Prop :=
  -- Arithmetic progression condition
  (15 : ℝ) - a = a - b ∧ a - b = b - (a * b) ∧
  -- Geometric progression condition
  ∃ r : ℝ, a * b = 15 * r^3 ∧ r > 0

/-- Theorem stating that (15, 15) is the only pair satisfying both conditions -/
theorem unique_satisfying_pair :
  ∀ a b : ℝ, SatisfyingPair a b → a = 15 ∧ b = 15 :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_pair_l1989_198978


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l1989_198959

/-- Represents the fraction of water remaining after a number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (replacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ replacements

/-- Proves that the fraction of water in the radiator after 5 replacements is 243/1024 -/
theorem radiator_water_fraction :
  waterFraction 20 5 5 = 243 / 1024 := by
  sorry

#eval waterFraction 20 5 5

end NUMINAMATH_CALUDE_radiator_water_fraction_l1989_198959


namespace NUMINAMATH_CALUDE_teal_survey_l1989_198908

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_green : more_green = 90)
  (h_both : both = 40)
  (h_neither : neither = 25) :
  total - (more_green - both + both + neither) = 75 :=
sorry

end NUMINAMATH_CALUDE_teal_survey_l1989_198908


namespace NUMINAMATH_CALUDE_denominator_of_0_34_l1989_198917

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem denominator_of_0_34 :
  (decimal_to_fraction 0.34).2 = 100 := by sorry

end NUMINAMATH_CALUDE_denominator_of_0_34_l1989_198917


namespace NUMINAMATH_CALUDE_simplify_fraction_l1989_198916

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1989_198916


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1989_198936

theorem pie_eating_contest (a b c d : ℤ) (h1 : a = 7 ∧ b = 8) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b - (c : ℚ) / d = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1989_198936


namespace NUMINAMATH_CALUDE_log_cos_sum_squared_l1989_198980

theorem log_cos_sum_squared : 
  (Real.log (Real.cos (20 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (40 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (80 * π / 180)) / Real.log (Real.sqrt 2)) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_cos_sum_squared_l1989_198980


namespace NUMINAMATH_CALUDE_second_person_receives_345_l1989_198996

/-- The total amount of money distributed -/
def total_amount : ℕ := 1000

/-- The sequence of distributions -/
def distribution_sequence (n : ℕ) : ℕ := n

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The largest n such that the sum of the first n natural numbers is at most the total amount -/
def max_n : ℕ := 44

/-- The amount received by the second person (Bernardo) -/
def amount_received_by_second : ℕ := 345

/-- Theorem stating that the second person (Bernardo) receives 345 reais -/
theorem second_person_receives_345 :
  (∀ n : ℕ, n ≤ max_n → sum_first_n n ≤ total_amount) →
  (∀ k : ℕ, k ≤ 15 → distribution_sequence (3*k - 1) ≤ max_n) →
  amount_received_by_second = 345 := by
  sorry

end NUMINAMATH_CALUDE_second_person_receives_345_l1989_198996


namespace NUMINAMATH_CALUDE_sum_square_value_l1989_198999

theorem sum_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 36) 
  (h2 : y * (x + y) = 72) : 
  (x + y)^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_sum_square_value_l1989_198999


namespace NUMINAMATH_CALUDE_flensburgian_iff_even_l1989_198993

/-- A set of equations is Flensburgian if there exists an i ∈ {1, 2, 3} such that
    for every solution where all variables are pairwise different, x_i > x_j for all j ≠ i -/
def IsFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, f x y z → x ≠ y → y ≠ z → z ≠ x →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The set of equations a^n + b = a and c^(n+1) + b^2 = ab -/
def EquationSet (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a * b

theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  IsFlensburgian (EquationSet n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flensburgian_iff_even_l1989_198993


namespace NUMINAMATH_CALUDE_clock_digit_sum_probability_l1989_198930

def total_times : ℕ := 1440
def times_with_sum_23 : ℕ := 4

theorem clock_digit_sum_probability :
  (times_with_sum_23 : ℚ) / total_times = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_clock_digit_sum_probability_l1989_198930


namespace NUMINAMATH_CALUDE_thomas_salary_l1989_198985

/-- Given the average salaries of two groups, prove Thomas's salary -/
theorem thomas_salary (raj_salary roshan_salary thomas_salary : ℕ) : 
  (raj_salary + roshan_salary) / 2 = 4000 →
  (raj_salary + roshan_salary + thomas_salary) / 3 = 5000 →
  thomas_salary = 7000 := by
  sorry

end NUMINAMATH_CALUDE_thomas_salary_l1989_198985


namespace NUMINAMATH_CALUDE_gcd_102_238_l1989_198957

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1989_198957


namespace NUMINAMATH_CALUDE_inequality_property_l1989_198921

theorem inequality_property (a b c d : ℝ) : a > b → c > d → a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l1989_198921


namespace NUMINAMATH_CALUDE_triangle_with_60_degree_angle_l1989_198937

/-- In a triangle with sides 4, 2√3, and 2 + 2√2, one of the angles is 60°. -/
theorem triangle_with_60_degree_angle :
  ∃ (a b c : ℝ) (α β γ : ℝ),
    a = 4 ∧ 
    b = 2 * Real.sqrt 3 ∧ 
    c = 2 + 2 * Real.sqrt 2 ∧
    α + β + γ = π ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos β ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧
    β = π/3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_with_60_degree_angle_l1989_198937


namespace NUMINAMATH_CALUDE_distance_to_line_not_greater_than_two_l1989_198984

/-- A structure representing a line in a plane -/
structure Line :=
  (points : Set Point)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The distance from a point to a line -/
def distanceToLine (p : Point) (l : Line) : ℝ := sorry

/-- Theorem: If a point P is outside a line l, and there are three points A, B, and C on l
    such that PA = 2, PB = 2.5, and PC = 3, then the distance from P to l is not greater than 2 -/
theorem distance_to_line_not_greater_than_two
  (P : Point) (l : Line) (A B C : Point)
  (h_P_outside : P ∉ l.points)
  (h_ABC_on_l : A ∈ l.points ∧ B ∈ l.points ∧ C ∈ l.points)
  (h_PA : distance P A = 2)
  (h_PB : distance P B = 2.5)
  (h_PC : distance P C = 3) :
  distanceToLine P l ≤ 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_not_greater_than_two_l1989_198984


namespace NUMINAMATH_CALUDE_runners_meet_time_l1989_198901

/-- The time in seconds for runner P to complete one round -/
def P_time : ℕ := 252

/-- The time in seconds for runner Q to complete one round -/
def Q_time : ℕ := 198

/-- The time in seconds for runner R to complete one round -/
def R_time : ℕ := 315

/-- The time after which all runners meet at the starting point -/
def meet_time : ℕ := 13860

/-- Theorem stating that the meet time is the least common multiple of individual round times -/
theorem runners_meet_time : 
  Nat.lcm (Nat.lcm P_time Q_time) R_time = meet_time := by sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1989_198901


namespace NUMINAMATH_CALUDE_households_with_car_l1989_198962

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end NUMINAMATH_CALUDE_households_with_car_l1989_198962


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1989_198986

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 6 / Real.sqrt 8) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1989_198986


namespace NUMINAMATH_CALUDE_park_visitors_l1989_198914

theorem park_visitors (visitors_day1 visitors_day2 : ℕ) : 
  visitors_day2 = visitors_day1 + 40 →
  visitors_day1 + visitors_day2 = 440 →
  visitors_day1 = 200 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l1989_198914


namespace NUMINAMATH_CALUDE_p_minus_q_value_l1989_198933

theorem p_minus_q_value (p q : ℚ) 
  (h1 : -6 / p = 3/2) 
  (h2 : 8 / q = -1/4) : 
  p - q = 28 := by
sorry

end NUMINAMATH_CALUDE_p_minus_q_value_l1989_198933


namespace NUMINAMATH_CALUDE_sibling_discount_calculation_l1989_198918

/-- Represents the tuition cost at the music school -/
def regular_tuition : ℕ := 45

/-- Represents the discounted cost for both children -/
def discounted_cost : ℕ := 75

/-- Represents the number of children -/
def num_children : ℕ := 2

/-- Calculates the sibling discount -/
def sibling_discount : ℕ :=
  regular_tuition * num_children - discounted_cost

theorem sibling_discount_calculation :
  sibling_discount = 15 := by sorry

end NUMINAMATH_CALUDE_sibling_discount_calculation_l1989_198918


namespace NUMINAMATH_CALUDE_max_profit_at_half_l1989_198928

/-- The profit function for a souvenir sale after process improvement -/
def profit_function (x : ℝ) : ℝ := 500 * (1 + 4*x - x^2 - 4*x^3)

/-- The theorem stating the maximum profit and the corresponding price increase -/
theorem max_profit_at_half :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 0 < x → x < 1 → profit_function x ≤ max_profit) ∧
    profit_function (1/2) = max_profit ∧
    max_profit = 11125 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_half_l1989_198928


namespace NUMINAMATH_CALUDE_sum_digits_base7_999_l1989_198991

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base7_999 : sumList (toBase7 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base7_999_l1989_198991


namespace NUMINAMATH_CALUDE_dispersion_measures_l1989_198973

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the statistics
def standardDeviation (s : Sample) : Real := sorry
def median (s : Sample) : Real := sorry
def range (s : Sample) : Real := sorry
def mean (s : Sample) : Real := sorry

-- Define a measure of dispersion
def measuresDispersion (f : Sample → Real) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  (measuresDispersion (standardDeviation)) ∧
  (measuresDispersion (range)) ∧
  (¬ measuresDispersion (median)) ∧
  (¬ measuresDispersion (mean)) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l1989_198973


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l1989_198963

/-- Calculates the number of granola bars per kid given the number of kids, bars per box, and boxes purchased. -/
def granola_bars_per_kid (num_kids : ℕ) (bars_per_box : ℕ) (boxes_purchased : ℕ) : ℕ :=
  (bars_per_box * boxes_purchased) / num_kids

/-- Proves that given 30 kids, 12 bars per box, and 5 boxes purchased, the number of granola bars per kid is 2. -/
theorem granola_bar_distribution : granola_bars_per_kid 30 12 5 = 2 := by
  sorry

#eval granola_bars_per_kid 30 12 5

end NUMINAMATH_CALUDE_granola_bar_distribution_l1989_198963


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1989_198958

def lowest_price : ℝ := 18
def highest_price : ℝ := 24

theorem price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 33.33333333333333 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1989_198958


namespace NUMINAMATH_CALUDE_profit_increase_calculation_l1989_198979

/-- Proves that given a 40% increase followed by a 20% decrease, 
    a final increase that results in an overall 68% increase must be a 50% increase. -/
theorem profit_increase_calculation (P : ℝ) (h : P > 0) : 
  let april_profit := 1.40 * P
  let may_profit := 0.80 * april_profit
  let june_profit := 1.68 * P
  (june_profit / may_profit - 1) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_calculation_l1989_198979


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1989_198987

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1989_198987


namespace NUMINAMATH_CALUDE_fifteen_customers_tipped_l1989_198925

/-- Calculates the number of customers who left a tip --/
def customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : ℕ :=
  initial_customers + additional_customers - non_tipping_customers

/-- Theorem: Given the conditions, prove that 15 customers left a tip --/
theorem fifteen_customers_tipped :
  customers_who_tipped 29 20 34 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_customers_tipped_l1989_198925


namespace NUMINAMATH_CALUDE_root_squared_plus_root_plus_one_equals_two_l1989_198904

theorem root_squared_plus_root_plus_one_equals_two (a : ℝ) : 
  a^2 + a - 1 = 0 → a^2 + a + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_squared_plus_root_plus_one_equals_two_l1989_198904
