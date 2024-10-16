import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_four_l41_4194

theorem arithmetic_sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_four_l41_4194


namespace NUMINAMATH_CALUDE_product_of_numbers_l41_4180

theorem product_of_numbers (x y : ℝ) : 
  x + y = 24 → x^2 + y^2 = 404 → x * y = 86 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l41_4180


namespace NUMINAMATH_CALUDE_welcoming_and_planning_committees_l41_4128

theorem welcoming_and_planning_committees (n : ℕ) : 
  (Nat.choose n 2 = 6) → (Nat.choose n 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_welcoming_and_planning_committees_l41_4128


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l41_4161

/-- Given a salt solution that is diluted, proves the initial salt concentration --/
theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 90)
  (h2 : water_added = 30)
  (h3 : final_concentration = 0.15)
  : ∃ (initial_concentration : ℝ),
    initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) ∧
    initial_concentration = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l41_4161


namespace NUMINAMATH_CALUDE_parabola_c_value_l41_4143

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := λ x => x^2 + b*x + c
  point1 : eq 2 = 12
  point2 : eq (-2) = 0
  point3 : eq 4 = 40

/-- The value of c for the parabola passing through (2, 12), (-2, 0), and (4, 40) is 2 -/
theorem parabola_c_value (p : Parabola) : p.c = 2 := by
  sorry

#check parabola_c_value

end NUMINAMATH_CALUDE_parabola_c_value_l41_4143


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_one_fifth_l41_4108

/-- Given that the terminal side of angle α passes through the point (3a, -4a) where a < 0,
    prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_equals_one_fifth 
  (α : Real) (a : Real) (h1 : a < 0) 
  (h2 : ∃ (t : Real), t > 0 ∧ Real.cos α = 3 * a / t ∧ Real.sin α = -4 * a / t) : 
  Real.sin α + Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_one_fifth_l41_4108


namespace NUMINAMATH_CALUDE_hyperbola_equation_l41_4164

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_asymptote : b / a = Real.sqrt 5 / 2
  h_shared_focus : ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 3

/-- The equation of the hyperbola is x^2/4 - y^2/5 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 4 ∧ C.b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l41_4164


namespace NUMINAMATH_CALUDE_bianca_deleted_pictures_l41_4177

theorem bianca_deleted_pictures (total_files songs text_files : ℕ) 
  (h1 : total_files = 17)
  (h2 : songs = 8)
  (h3 : text_files = 7)
  : total_files = songs + text_files + 2 := by
  sorry

end NUMINAMATH_CALUDE_bianca_deleted_pictures_l41_4177


namespace NUMINAMATH_CALUDE_fair_coin_probability_l41_4107

/-- A coin toss with two possible outcomes -/
inductive CoinOutcome
  | heads
  | tails

/-- The probability of a coin toss outcome -/
def probability (outcome : CoinOutcome) : Real :=
  0.5

/-- Theorem stating that the probability of getting heads or tails is 0.5 -/
theorem fair_coin_probability :
  ∀ (outcome : CoinOutcome), probability outcome = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l41_4107


namespace NUMINAMATH_CALUDE_equal_count_for_any_number_l41_4132

/-- A function that represents the number of n-digit numbers from which 
    a k-digit number composed of only 1 and 2 can be obtained by erasing digits -/
def F (k n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is composed only of digits 1 and 2 -/
def OnlyOneTwo (x : ℕ) : Prop := sorry

theorem equal_count_for_any_number (k n : ℕ) (X Y : ℕ) (h1 : k > 0) (h2 : n ≥ k) 
  (hX : OnlyOneTwo X) (hY : OnlyOneTwo Y) 
  (hXdigits : X < 10^k) (hYdigits : Y < 10^k) : F k n = F k n := by
  sorry

end NUMINAMATH_CALUDE_equal_count_for_any_number_l41_4132


namespace NUMINAMATH_CALUDE_horner_method_v3_l41_4181

def horner_polynomial (x : ℚ) : ℚ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (x : ℚ) : ℚ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

theorem horner_method_v3 :
  horner_v3 (-4) = -49 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l41_4181


namespace NUMINAMATH_CALUDE_three_primes_in_list_l41_4140

def number_list : List Nat := [11, 12, 13, 14, 15, 16, 17]

theorem three_primes_in_list :
  (number_list.filter Nat.Prime).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_primes_in_list_l41_4140


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l41_4174

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ (∃ x : ℝ, x^2 + 1 < x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l41_4174


namespace NUMINAMATH_CALUDE_three_solutions_sum_l41_4188

theorem three_solutions_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_solutions : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ = b ∧
    (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  a + b = 144 := by
sorry

end NUMINAMATH_CALUDE_three_solutions_sum_l41_4188


namespace NUMINAMATH_CALUDE_basketball_price_proof_l41_4137

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 124

/-- The price of a soccer ball in yuan -/
def soccer_ball_price : ℕ := 62

/-- The total cost of a basketball and a soccer ball in yuan -/
def total_cost : ℕ := 186

theorem basketball_price_proof :
  (basketball_price = 124) ∧
  (basketball_price + soccer_ball_price = total_cost) ∧
  (basketball_price = 2 * soccer_ball_price) :=
sorry

end NUMINAMATH_CALUDE_basketball_price_proof_l41_4137


namespace NUMINAMATH_CALUDE_square_root_of_1_5625_l41_4154

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1_5625_l41_4154


namespace NUMINAMATH_CALUDE_union_of_sets_l41_4173

theorem union_of_sets : 
  let A : Set Int := {-1, 1, 2, 4}
  let B : Set Int := {-1, 0, 2}
  A ∪ B = {-1, 0, 1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l41_4173


namespace NUMINAMATH_CALUDE_martha_cakes_l41_4187

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) (h1 : num_children = 3) (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 :=
by sorry

end NUMINAMATH_CALUDE_martha_cakes_l41_4187


namespace NUMINAMATH_CALUDE_not_all_on_curve_implies_exists_off_curve_l41_4145

-- Define the necessary types and functions
variable (X Y : Type) -- X and Y represent coordinate types
variable (C : Set (X × Y)) -- C represents the curve
variable (f : X → Y → Prop) -- f represents the equation f(x, y) = 0

-- The main theorem
theorem not_all_on_curve_implies_exists_off_curve :
  (¬ ∀ x y, f x y → (x, y) ∈ C) →
  ∃ x y, f x y ∧ (x, y) ∉ C := by
sorry

end NUMINAMATH_CALUDE_not_all_on_curve_implies_exists_off_curve_l41_4145


namespace NUMINAMATH_CALUDE_solve_equation_l41_4102

-- Define a custom pair type for real numbers
structure RealPair :=
  (fst : ℝ)
  (snd : ℝ)

-- Define equality for RealPair
def realPairEq (a b : RealPair) : Prop :=
  a.fst = b.fst ∧ a.snd = b.snd

-- Define the ⊕ operation
def oplus (a b : RealPair) : RealPair :=
  ⟨a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst⟩

-- Theorem statement
theorem solve_equation (p q : ℝ) :
  oplus ⟨1, 2⟩ ⟨p, q⟩ = ⟨5, 0⟩ → realPairEq ⟨p, q⟩ ⟨1, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l41_4102


namespace NUMINAMATH_CALUDE_sum_of_squares_real_roots_l41_4156

theorem sum_of_squares_real_roots (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^4 + a*x₁^2 - 2017 = 0) ∧ 
    (x₂^4 + a*x₂^2 - 2017 = 0) ∧ 
    (x₃^4 + a*x₃^2 - 2017 = 0) ∧ 
    (x₄^4 + a*x₄^2 - 2017 = 0) ∧ 
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4)) → 
  a = 1006.5 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_real_roots_l41_4156


namespace NUMINAMATH_CALUDE_hyperbola_equation_l41_4142

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = √3 x and one of its foci lies on the line x = -6,
    then the equation of the hyperbola is x²/9 - y²/27 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 6) →
  b/a = Real.sqrt 3 →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/27 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l41_4142


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l41_4199

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x * y + x * z + y * z ≠ 0) :
  (x^3 + y^3 + z^3) / (x * y * z * (x * y + x * z + y * z)) = -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l41_4199


namespace NUMINAMATH_CALUDE_f_properties_l41_4148

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

theorem f_properties (a : ℝ) :
  /- 1. Tangent line equation when a = 3 at x = 1 -/
  (a = 3 → ∃ (m b : ℝ), m = 3 ∧ b = -5 ∧ ∀ x y, y = f 3 x ↔ m*x - y - b = 0) ∧
  /- 2. Monotonicity depends on a -/
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ f a x1 > f a x2) ∧
  (∃ (x3 x4 : ℝ), x3 < x4 ∧ f a x3 < f a x4) ∧
  /- 3. Condition for f(x0) > 0 -/
  (∃ (x0 : ℝ), x0 > 0 ∧ f a x0 > 0) ↔ a > 3 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l41_4148


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l41_4186

def total_amount : ℝ := 320
def amount_left : ℝ := 224

theorem sandy_spending_percentage :
  (total_amount - amount_left) / total_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l41_4186


namespace NUMINAMATH_CALUDE_invisible_dots_count_l41_4121

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the numbers on a single die
def die_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the visible numbers
def visible_numbers : List ℕ := [2, 2, 3, 4, 4, 5, 6, 6]

-- Theorem to prove
theorem invisible_dots_count :
  (num_dice * (die_numbers.sum)) - (visible_numbers.sum) = 52 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l41_4121


namespace NUMINAMATH_CALUDE_max_sum_of_squared_distances_l41_4115

variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_distances_l41_4115


namespace NUMINAMATH_CALUDE_unique_root_range_l41_4192

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * Real.exp x else -Real.log x

theorem unique_root_range (a : ℝ) :
  (∃! x, f a (f a x) = 0) → a ∈ Set.Ioi 0 ∪ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_unique_root_range_l41_4192


namespace NUMINAMATH_CALUDE_tristan_saturday_study_time_l41_4131

/-- Calculates Tristan's study hours on Saturday given his study schedule --/
def tristanSaturdayStudyHours (mondayHours : ℕ) (weekdayHours : ℕ) (totalWeekHours : ℕ) : ℕ :=
  let tuesdayHours := 2 * mondayHours
  let wednesdayToFridayHours := 3 * weekdayHours
  let mondayToFridayHours := mondayHours + tuesdayHours + wednesdayToFridayHours
  let remainingHours := totalWeekHours - mondayToFridayHours
  remainingHours / 2

/-- Theorem: Given Tristan's study schedule, he studies for 2 hours on Saturday --/
theorem tristan_saturday_study_time :
  tristanSaturdayStudyHours 4 3 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tristan_saturday_study_time_l41_4131


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l41_4111

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + initial_height * rebound_ratio + initial_height * rebound_ratio

/-- Theorem: A ball dropped from 100 cm with 50% rebound travels 200 cm when it touches the floor the third time -/
theorem bouncing_ball_distance :
  total_distance 100 0.5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l41_4111


namespace NUMINAMATH_CALUDE_two_hundred_thousand_squared_l41_4197

theorem two_hundred_thousand_squared : 200000 * 200000 = 40000000000 := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_thousand_squared_l41_4197


namespace NUMINAMATH_CALUDE_percent_of_decimal_zero_point_zero_one_is_ten_percent_of_zero_point_one_l41_4124

theorem percent_of_decimal (x y : ℝ) (h : y ≠ 0) :
  x / y * 100 = (x / y * 100 : ℝ) :=
by sorry

theorem zero_point_zero_one_is_ten_percent_of_zero_point_one :
  (0.01 : ℝ) / 0.1 * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_percent_of_decimal_zero_point_zero_one_is_ten_percent_of_zero_point_one_l41_4124


namespace NUMINAMATH_CALUDE_negation_equivalence_l41_4101

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l41_4101


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l41_4112

theorem complex_exponential_sum : 
  12 * Complex.exp (Complex.I * Real.pi / 7) + 12 * Complex.exp (Complex.I * 19 * Real.pi / 14) = 
  24 * Real.cos (5 * Real.pi / 28) * Complex.exp (Complex.I * 3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l41_4112


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l41_4113

/-- The probability of getting at least one head when tossing four fair coins -/
theorem prob_at_least_one_head_four_coins : ℝ :=
  let p_tail : ℝ := 1 / 2  -- probability of getting a tail on one coin toss
  let p_all_tails : ℝ := p_tail ^ 4  -- probability of getting all tails
  1 - p_all_tails

/-- Proof that the probability of getting at least one head when tossing four fair coins is 15/16 -/
theorem prob_at_least_one_head_four_coins_is_15_16 :
  prob_at_least_one_head_four_coins = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l41_4113


namespace NUMINAMATH_CALUDE_bridge_distance_l41_4179

theorem bridge_distance (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (O₁O₂ - 5.75) < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_distance_l41_4179


namespace NUMINAMATH_CALUDE_teacher_age_l41_4172

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 19 →
  student_avg_age = 20 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l41_4172


namespace NUMINAMATH_CALUDE_scientific_notation_of_120000_l41_4106

theorem scientific_notation_of_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120000_l41_4106


namespace NUMINAMATH_CALUDE_inequality_sum_l41_4175

theorem inequality_sum (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) (h6 : d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l41_4175


namespace NUMINAMATH_CALUDE_special_function_properties_l41_4116

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l41_4116


namespace NUMINAMATH_CALUDE_rectangular_prism_edge_pairs_l41_4155

/-- A rectangular prism -/
structure RectangularPrism where
  -- We don't need to define the actual structure, just its existence

/-- Count of parallel edge pairs in a rectangular prism -/
def parallel_edge_pairs (rp : RectangularPrism) : ℕ := sorry

/-- Count of perpendicular edge pairs in a rectangular prism -/
def perpendicular_edge_pairs (rp : RectangularPrism) : ℕ := sorry

/-- Theorem stating the counts of parallel and perpendicular edge pairs in a rectangular prism -/
theorem rectangular_prism_edge_pairs (rp : RectangularPrism) :
  parallel_edge_pairs rp = 8 ∧ perpendicular_edge_pairs rp = 20 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_edge_pairs_l41_4155


namespace NUMINAMATH_CALUDE_second_hand_store_shirt_price_l41_4139

/-- The price of a shirt sold to the second-hand store -/
def shirt_price : ℚ := 4

theorem second_hand_store_shirt_price :
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_sold : ℕ := 5
  let pants_price : ℚ := 5
  let shorts_price : ℚ := 3
  let new_shirts_bought : ℕ := 2
  let new_shirt_price : ℚ := 10
  let remaining_money : ℚ := 30

  shirt_price * shirts_sold + 
  pants_price * pants_sold + 
  shorts_price * shorts_sold = 
  remaining_money + new_shirt_price * new_shirts_bought := by sorry

end NUMINAMATH_CALUDE_second_hand_store_shirt_price_l41_4139


namespace NUMINAMATH_CALUDE_g_53_l41_4120

/-- A function satisfying g(xy) = yg(x) for all real x and y, with g(1) = 15 -/
def g : ℝ → ℝ :=
  sorry

/-- The functional equation for g -/
axiom g_eq (x y : ℝ) : g (x * y) = y * g x

/-- The value of g at 1 -/
axiom g_one : g 1 = 15

/-- The theorem to be proved -/
theorem g_53 : g 53 = 795 :=
  sorry

end NUMINAMATH_CALUDE_g_53_l41_4120


namespace NUMINAMATH_CALUDE_union_of_intervals_l41_4165

open Set

theorem union_of_intervals (A B : Set ℝ) : 
  A = {x : ℝ | 3 < x ∧ x ≤ 7} →
  B = {x : ℝ | 4 < x ∧ x ≤ 10} →
  A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} :=
by sorry

end NUMINAMATH_CALUDE_union_of_intervals_l41_4165


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l41_4105

/-- For a geometric sequence with common ratio -1/3, the ratio of the sum of odd-indexed terms
    to the sum of even-indexed terms (up to the 8th term) is -3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : q = -1/3) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l41_4105


namespace NUMINAMATH_CALUDE_grid_completion_count_l41_4130

/-- Represents a 2x3 grid with one fixed R and 5 remaining squares --/
def Grid := Fin 2 → Fin 3 → Fin 3

/-- Checks if two adjacent cells in the grid have the same value --/
def has_adjacent_match (g : Grid) : Prop :=
  ∃ i j, (g i j = g i (j + 1)) ∨ 
         (g i j = g (i + 1) j)

/-- The number of ways to fill the grid --/
def total_configurations : ℕ := 3^5

/-- The number of valid configurations without adjacent matches --/
def valid_configurations : ℕ := 18

theorem grid_completion_count :
  (total_configurations - valid_configurations : ℕ) = 225 :=
sorry

end NUMINAMATH_CALUDE_grid_completion_count_l41_4130


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_l41_4144

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_l41_4144


namespace NUMINAMATH_CALUDE_unique_root_quadratic_root_l41_4151

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  has_unique_root : (b ^ 2 - 4 * a * c) = 0

/-- The theorem stating that the root of the quadratic polynomial is -11 -/
theorem unique_root_quadratic_root (f : UniqueRootQuadratic) 
  (h : ∃ g : UniqueRootQuadratic, 
    g.a = -f.a ∧ 
    g.b = (f.b - 30 * f.a) ∧ 
    g.c = (17 * f.a - 7 * f.b + f.c)) :
  (f.a ≠ 0) → (-f.b / (2 * f.a)) = -11 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_root_l41_4151


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l41_4157

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  (1 / (1 + a) + 4 / (4 + b)) ≥ 9/8 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 3 ∧ 1 / (1 + a₀) + 4 / (4 + b₀) = 9/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l41_4157


namespace NUMINAMATH_CALUDE_solution_value_l41_4149

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ (m / (2 - x)) - (1 / (x - 2)) = 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l41_4149


namespace NUMINAMATH_CALUDE_flowerbed_width_l41_4114

theorem flowerbed_width (width length perimeter : ℝ) : 
  width > 0 →
  length > 0 →
  length = 2 * width - 1 →
  perimeter = 2 * length + 2 * width →
  perimeter = 22 →
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_flowerbed_width_l41_4114


namespace NUMINAMATH_CALUDE_smallest_sum_c_d_l41_4135

theorem smallest_sum_c_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (192/81)^(1/3) + (12 * (192/81)^(1/3))^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_c_d_l41_4135


namespace NUMINAMATH_CALUDE_value_is_square_of_number_l41_4141

theorem value_is_square_of_number (n v : ℕ) : 
  n = 14 → 
  v = n^2 → 
  n + v = 210 → 
  v = 196 := by sorry

end NUMINAMATH_CALUDE_value_is_square_of_number_l41_4141


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l41_4163

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a < 10 ∧ n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3)

def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let rest := n % 100
  d2 * 1000 + d1 * 100 + rest

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem unique_four_digit_square : 
  ∀ (n : ℕ), 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ 
     is_perfect_square (swap_first_two_digits n)) ↔ 
    n = 4356 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l41_4163


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l41_4122

theorem complex_fraction_equality : (3 + I) / (1 + I) = 2 - I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l41_4122


namespace NUMINAMATH_CALUDE_right_triangle_special_property_l41_4167

theorem right_triangle_special_property :
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive sides
  (a^2 + b^2 = c^2) →        -- right triangle (Pythagorean theorem)
  ((1/2) * a * b = 24) →     -- area is 24
  (a^2 + b^2 = 2 * 24) →     -- sum of squares of legs equals twice the area
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_special_property_l41_4167


namespace NUMINAMATH_CALUDE_consecutive_numbers_equation_l41_4190

theorem consecutive_numbers_equation :
  ∃ (a b c d : ℕ), (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (a * c - b * d = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_equation_l41_4190


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l41_4138

theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = 2/5) :
  (2*a + b)^2 - (3*b + 2*a) * (2*a - 3*b) = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l41_4138


namespace NUMINAMATH_CALUDE_tv_diagonal_problem_l41_4109

theorem tv_diagonal_problem (larger_diagonal smaller_diagonal : ℝ) :
  larger_diagonal = 24 →
  larger_diagonal ^ 2 / 2 - smaller_diagonal ^ 2 / 2 = 143.5 →
  smaller_diagonal = 17 := by
sorry

end NUMINAMATH_CALUDE_tv_diagonal_problem_l41_4109


namespace NUMINAMATH_CALUDE_quadratic_expression_sum_l41_4191

theorem quadratic_expression_sum (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d^2 + b * d + c ∧ a + b + c = 53 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_sum_l41_4191


namespace NUMINAMATH_CALUDE_seating_theorem_l41_4189

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (abc : ℕ) (de : ℕ) : ℕ :=
  factorial n - (factorial (n - 2) * factorial 3) - 
  (factorial (n - 1) * factorial 2) + 
  (factorial (n - 3) * factorial 3 * factorial 2)

theorem seating_theorem : 
  seating_arrangements 10 3 2 = 2853600 := by sorry

end NUMINAMATH_CALUDE_seating_theorem_l41_4189


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l41_4103

/-- Given a circle with equation x^2 + y^2 - 2x = 0, its center is (1,0) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l41_4103


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l41_4153

theorem yellow_marbles_count (total : ℕ) (red : ℕ) 
  (h1 : total = 140)
  (h2 : red = 10)
  (h3 : ∃ blue : ℕ, blue = (5 * red) / 2)
  (h4 : ∃ green : ℕ, green = ((13 * blue) / 10))
  (h5 : ∃ yellow : ℕ, yellow = total - (blue + red + green)) :
  yellow = 73 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l41_4153


namespace NUMINAMATH_CALUDE_fruit_store_discount_l41_4127

/-- Represents the discount policy of a fruit store -/
def discount_policy (lemon_price papaya_price mango_price : ℕ)
                    (lemon_qty papaya_qty mango_qty : ℕ)
                    (total_paid : ℕ) : ℕ :=
  let total_cost := lemon_price * lemon_qty + papaya_price * papaya_qty + mango_price * mango_qty
  let total_fruits := lemon_qty + papaya_qty + mango_qty
  total_cost - total_paid

theorem fruit_store_discount :
  discount_policy 2 1 4 6 4 2 21 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_store_discount_l41_4127


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l41_4171

/-- Given a monotonically decreasing exponential function f(x) = a^x on ℝ,
    prove that when f(x+1) ≥ 1, the range of x is (-∞, -1]. -/
theorem exponential_decreasing_range (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a^x) :
  (∀ x y, x < y → f x > f y) →
  {x : ℝ | f (x + 1) ≥ 1} = Set.Iic (-1) := by
sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l41_4171


namespace NUMINAMATH_CALUDE_negation_of_proposition_l41_4136

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x > 0, (x + 1) * Real.exp x > 1) ↔ (∃ x₀ > 0, (x₀ + 1) * Real.exp x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l41_4136


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l41_4166

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l41_4166


namespace NUMINAMATH_CALUDE_parabola_coefficients_l41_4150

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, parabola a b c x = parabola a b c (4 - x)) ∧  -- Vertical axis of symmetry at x = 2
    (parabola a b c 2 = 3) ∧                            -- Vertex at (2, 3)
    (parabola a b c 0 = 1) ∧                            -- Passes through (0, 1)
    (a = -1/2 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l41_4150


namespace NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l41_4123

theorem eighteen_percent_of_500_is_90 (x : ℝ) : 
  (18 / 100) * x = 90 → x = 500 := by sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l41_4123


namespace NUMINAMATH_CALUDE_sheets_per_box_l41_4160

theorem sheets_per_box (total_sheets : ℕ) (num_boxes : ℕ) (h1 : total_sheets = 700) (h2 : num_boxes = 7) :
  total_sheets / num_boxes = 100 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_box_l41_4160


namespace NUMINAMATH_CALUDE_divisors_of_360_l41_4195

theorem divisors_of_360 : ∃ (d : Finset Nat), 
  (∀ x ∈ d, x ∣ 360) ∧ 
  (∀ x : Nat, x ∣ 360 → x ∈ d) ∧
  d.card = 24 ∧
  d.sum id = 1170 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_360_l41_4195


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l41_4182

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℤ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l41_4182


namespace NUMINAMATH_CALUDE_line_inclination_angle_l41_4117

/-- The inclination angle of a line with equation √3x - y + 1 = 0 is 60°. -/
theorem line_inclination_angle (x y : ℝ) :
  (Real.sqrt 3 * x - y + 1 = 0) → (∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l41_4117


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l41_4158

/-- A function f is decreasing on ℝ -/
def DecreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a decreasing function f on ℝ, if f(m-1) > f(2m-1), then m > 0 -/
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnReals f) (h_inequality : f (m - 1) > f (2 * m - 1)) : 
  m > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l41_4158


namespace NUMINAMATH_CALUDE_dans_age_l41_4118

theorem dans_age : ∃ x : ℕ, (x + 18 = 5 * (x - 6)) ∧ (x = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_dans_age_l41_4118


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l41_4196

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = -4) :
  (x^2 / (x - 1) - x + 1) / ((4 * x^2 - 4 * x + 1) / (1 - x)) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l41_4196


namespace NUMINAMATH_CALUDE_xingguang_pass_rate_l41_4146

/-- Calculates the pass rate for a physical fitness test -/
def pass_rate (total_students : ℕ) (failed_students : ℕ) : ℚ :=
  (total_students - failed_students : ℚ) / total_students * 100

/-- Theorem: The pass rate for Xingguang Primary School's physical fitness test is 92% -/
theorem xingguang_pass_rate :
  pass_rate 500 40 = 92 := by
  sorry

end NUMINAMATH_CALUDE_xingguang_pass_rate_l41_4146


namespace NUMINAMATH_CALUDE_max_regions_circular_disk_l41_4133

/-- 
Given a circular disk divided by 2n equally spaced radii (n > 0) and one chord,
the maximum number of non-overlapping regions is 3n + 1.
-/
theorem max_regions_circular_disk (n : ℕ) (h : n > 0) : 
  ∃ (num_regions : ℕ), num_regions = 3 * n + 1 ∧ 
  (∀ (m : ℕ), m ≤ num_regions) := by
sorry

end NUMINAMATH_CALUDE_max_regions_circular_disk_l41_4133


namespace NUMINAMATH_CALUDE_correct_initial_distribution_l41_4119

/-- Represents the initial and final coin counts for each person -/
structure CoinCounts where
  initial_gold : ℕ
  initial_silver : ℕ
  final_gold : ℕ
  final_silver : ℕ

/-- Represents the treasure distribution problem -/
def treasure_distribution (k : CoinCounts) (v : CoinCounts) : Prop :=
  -- Křemílek loses half of his gold coins
  k.initial_gold / 2 = k.final_gold - v.initial_gold / 3 ∧
  -- Vochomůrka loses half of his silver coins
  v.initial_silver / 2 = v.final_silver - k.initial_silver / 4 ∧
  -- Vochomůrka gives one-third of his remaining gold coins to Křemílek
  v.initial_gold * 2 / 3 = v.final_gold ∧
  -- Křemílek gives one-quarter of his silver coins to Vochomůrka
  k.initial_silver * 3 / 4 = k.final_silver ∧
  -- After exchanges, each has exactly 12 gold coins and 18 silver coins
  k.final_gold = 12 ∧ k.final_silver = 18 ∧
  v.final_gold = 12 ∧ v.final_silver = 18

/-- Theorem stating the correct initial distribution of coins -/
theorem correct_initial_distribution :
  ∃ (k v : CoinCounts),
    treasure_distribution k v ∧
    k.initial_gold = 12 ∧ k.initial_silver = 24 ∧
    v.initial_gold = 18 ∧ v.initial_silver = 24 :=
  sorry

end NUMINAMATH_CALUDE_correct_initial_distribution_l41_4119


namespace NUMINAMATH_CALUDE_opposite_to_gold_is_yellow_l41_4104

/-- Represents the colors used on the cube faces -/
inductive Color
  | Blue
  | Yellow
  | Orange
  | Black
  | Silver
  | Gold

/-- Represents the positions of faces on the cube -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube, showing top, front, and right faces -/
structure CubeView where
  top : Color
  front : Color
  right : Color

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Position → Color

/-- The three views of the cube given in the problem -/
def givenViews : List CubeView := [
  { top := Color.Blue, front := Color.Yellow, right := Color.Orange },
  { top := Color.Blue, front := Color.Black,  right := Color.Orange },
  { top := Color.Blue, front := Color.Silver, right := Color.Orange }
]

/-- Theorem stating that the face opposite to gold is yellow -/
theorem opposite_to_gold_is_yellow (cube : Cube) 
    (h1 : ∀ view ∈ givenViews, 
      cube.faces Position.Top = view.top ∧ 
      cube.faces Position.Right = view.right ∧ 
      (cube.faces Position.Front = view.front ∨ 
       cube.faces Position.Left = view.front ∨ 
       cube.faces Position.Bottom = view.front))
    (h2 : ∃! pos, cube.faces pos = Color.Gold) :
    cube.faces Position.Front = Color.Yellow :=
  sorry

end NUMINAMATH_CALUDE_opposite_to_gold_is_yellow_l41_4104


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l41_4198

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) :
  a * 1 + b * (-1) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l41_4198


namespace NUMINAMATH_CALUDE_right_triangle_sets_l41_4125

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 6 8 10) ∧
  ¬(is_right_triangle 5 8 13) ∧
  ¬(is_right_triangle 12 13 14) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l41_4125


namespace NUMINAMATH_CALUDE_digit_difference_of_63_l41_4126

theorem digit_difference_of_63 :
  let tens : ℕ := 63 / 10
  let ones : ℕ := 63 % 10
  tens + ones = 9 →
  tens - ones = 3 :=
by sorry

end NUMINAMATH_CALUDE_digit_difference_of_63_l41_4126


namespace NUMINAMATH_CALUDE_ratio_expression_value_l41_4129

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l41_4129


namespace NUMINAMATH_CALUDE_probability_of_different_digits_l41_4193

/-- The number of integers from 100 to 999 inclusive -/
def total_integers : ℕ := 999 - 100 + 1

/-- The number of integers from 100 to 999 with all different digits -/
def integers_with_different_digits : ℕ := 9 * 9 * 8

/-- The probability of selecting an integer with all different digits from 100 to 999 -/
def probability : ℚ := integers_with_different_digits / total_integers

theorem probability_of_different_digits : probability = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_different_digits_l41_4193


namespace NUMINAMATH_CALUDE_rose_count_l41_4176

theorem rose_count (lilies roses tulips : ℕ) : 
  roses = lilies + 22 →
  roses = tulips - 20 →
  lilies + roses + tulips = 100 →
  roses = 34 := by
sorry

end NUMINAMATH_CALUDE_rose_count_l41_4176


namespace NUMINAMATH_CALUDE_ellipse_equation_l41_4170

/-- Represents an ellipse with foci on coordinate axes and midpoint at origin -/
structure Ellipse where
  focal_distance : ℝ
  sum_distances : ℝ

/-- The equation of the ellipse when foci are on the x-axis -/
def ellipse_equation_x (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / ((e.sum_distances / 2)^2) + y^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- The equation of the ellipse when foci are on the y-axis -/
def ellipse_equation_y (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / ((e.sum_distances / 2)^2) + x^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation (e : Ellipse) (h1 : e.focal_distance = 8) (h2 : e.sum_distances = 12) :
  ∀ x y : ℝ, ellipse_equation_x e x y ∨ ellipse_equation_y e x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l41_4170


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l41_4152

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) ↔ a * x^2 + b * x + c ≥ 0) →
  (a > 0 ∧
   (∀ x : ℝ, bx + c > 0 ↔ x < -6) ∧
   a + b + c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l41_4152


namespace NUMINAMATH_CALUDE_paths_with_consecutive_right_moves_l41_4185

/-- The number of paths on a grid with specified conditions -/
def num_paths (horizontal_steps vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps - 1) vertical_steps

/-- The main theorem stating the number of paths under given conditions -/
theorem paths_with_consecutive_right_moves :
  num_paths 7 6 = 924 :=
by
  sorry

end NUMINAMATH_CALUDE_paths_with_consecutive_right_moves_l41_4185


namespace NUMINAMATH_CALUDE_inequality_proof_l41_4169

theorem inequality_proof (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l41_4169


namespace NUMINAMATH_CALUDE_haley_has_35_marbles_l41_4178

/-- The number of marbles Haley has, given the number of boys and marbles per boy -/
def haley_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley has 35 marbles -/
theorem haley_has_35_marbles :
  haley_marbles 5 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_has_35_marbles_l41_4178


namespace NUMINAMATH_CALUDE_distribution_count_l41_4159

def number_of_women : ℕ := 2
def number_of_men : ℕ := 10
def number_of_magazines : ℕ := 8
def number_of_newspapers : ℕ := 4

theorem distribution_count :
  (Nat.choose number_of_men (number_of_newspapers - 1)) +
  (Nat.choose number_of_men number_of_newspapers) = 255 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l41_4159


namespace NUMINAMATH_CALUDE_cross_number_intersection_l41_4147

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_three (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def power_of_seven (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem cross_number_intersection :
  ∃! d : ℕ,
    d < 10 ∧
    ∃ (n m : ℕ),
      is_three_digit n ∧
      is_three_digit m ∧
      power_of_three n ∧
      power_of_seven m ∧
      n % 10 = d ∧
      (m / 100) % 10 = d :=
sorry

end NUMINAMATH_CALUDE_cross_number_intersection_l41_4147


namespace NUMINAMATH_CALUDE_university_subjects_overlap_l41_4184

/-- The problem of students studying Physics and Chemistry at a university --/
theorem university_subjects_overlap (total : ℕ) (physics_min physics_max chem_min chem_max : ℕ) :
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chem_min = 1000 →
  chem_max = 1125 →
  let m := physics_min + chem_min - total
  let M := physics_max + chem_max - total
  M - m = 250 := by
  sorry

end NUMINAMATH_CALUDE_university_subjects_overlap_l41_4184


namespace NUMINAMATH_CALUDE_blue_balls_count_l41_4162

theorem blue_balls_count (total : ℕ) (p : ℚ) (h1 : total = 12) (h2 : p = 1 / 22) : 
  let red : ℕ → Prop := λ r => r * (r - 1) / (total * (total - 1)) = p
  ∃ r : ℕ, red r ∧ total - r = 9 :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l41_4162


namespace NUMINAMATH_CALUDE_f_negative_l41_4134

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ := x * (1 - x)

-- Theorem to prove
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = x * (1 + x) :=
by sorry

end NUMINAMATH_CALUDE_f_negative_l41_4134


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l41_4110

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_two : x + y + z = 2) :
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l41_4110


namespace NUMINAMATH_CALUDE_coefficient_of_y_is_one_l41_4168

/-- A line passing through two points with a given equation -/
structure Line where
  m : ℝ
  n : ℝ
  p : ℝ
  equation : ℝ → ℝ → Prop

/-- The line satisfies the given conditions -/
def line_satisfies_conditions (L : Line) : Prop :=
  L.p = 0.6666666666666666 ∧
  ∀ x y, L.equation x y ↔ x = y + 5

/-- The coefficient of y in the line equation is 1 -/
theorem coefficient_of_y_is_one (L : Line) 
  (h : line_satisfies_conditions L) : 
  ∃ b : ℝ, ∀ x y, L.equation x y ↔ y = x + b :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_is_one_l41_4168


namespace NUMINAMATH_CALUDE_parabola_equation_l41_4100

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

/-- Definition of the left vertex of the hyperbola -/
def left_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y = 0

/-- Definition of a parabola passing through a point -/
def parabola_through_point (eq : ℝ → ℝ → Prop) (px py : ℝ) : Prop :=
  eq px py

/-- Theorem stating the standard equation of the parabola -/
theorem parabola_equation (f : ℝ → ℝ → Prop) (fx fy : ℝ) :
  left_vertex fx fy →
  parabola_through_point f 2 (-4) →
  (∀ x y, f x y ↔ y^2 = 8*x) ∨ (∀ x y, f x y ↔ x^2 = -y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l41_4100


namespace NUMINAMATH_CALUDE_expression_simplification_l41_4183

theorem expression_simplification (x y : ℝ) 
  (h : |x + 1| + (2 * y - 4)^2 = 0) : 
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l41_4183
