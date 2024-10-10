import Mathlib

namespace simplify_expression_l29_2953

theorem simplify_expression (s : ℝ) : 105 * s - 63 * s = 42 * s := by
  sorry

end simplify_expression_l29_2953


namespace y_in_terms_of_x_l29_2967

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end y_in_terms_of_x_l29_2967


namespace modular_equivalence_123456_l29_2956

theorem modular_equivalence_123456 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end modular_equivalence_123456_l29_2956


namespace sum_congruence_l29_2968

theorem sum_congruence : (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_congruence_l29_2968


namespace extreme_values_of_f_l29_2988

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c) → c = a) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≥ f c) → c = b) :=
sorry

end extreme_values_of_f_l29_2988


namespace candies_remaining_l29_2923

/-- Calculates the number of candies remaining after Carlos ate all yellow candies -/
theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow = 3 * red - 20)
  (h3 : blue = yellow / 2) :
  red + blue = 90 := by
  sorry

#check candies_remaining

end candies_remaining_l29_2923


namespace five_digit_number_proof_l29_2934

theorem five_digit_number_proof (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 10 * x + 1 = 3 * (100000 + x) → x = 42857 := by
  sorry

end five_digit_number_proof_l29_2934


namespace ordered_pairs_count_l29_2963

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 6 / m + 3 / n = 1) ∧
    pairs.length = 6 := by
  sorry

end ordered_pairs_count_l29_2963


namespace max_discount_rate_l29_2962

/-- Represents the maximum discount rate problem --/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1) :
  ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (selling_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end max_discount_rate_l29_2962


namespace tan_theta_value_l29_2921

theorem tan_theta_value (θ : Real) (a : Real) 
  (h1 : (4, a) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ})
  (h2 : Real.sin (θ - π) = 3/5) : 
  Real.tan θ = -3/4 := by
sorry

end tan_theta_value_l29_2921


namespace min_colors_regular_ngon_l29_2924

/-- 
Represents a coloring of sides and diagonals in a regular n-gon.
The coloring is valid if any two segments sharing a common point have different colors.
-/
def ValidColoring (n : ℕ) := 
  { coloring : (Fin n × Fin n) → ℕ // 
    ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → 
    coloring (i, j) ≠ coloring (i, k) ∧ 
    coloring (i, j) ≠ coloring (j, k) ∧ 
    coloring (i, k) ≠ coloring (j, k) }

/-- 
The minimum number of colors needed for a valid coloring of a regular n-gon 
is equal to n.
-/
theorem min_colors_regular_ngon (n : ℕ) (h : n ≥ 3) : 
  (∃ (c : ValidColoring n), ∀ (i j : Fin n), c.val (i, j) < n) ∧ 
  (∀ (c : ValidColoring n) (m : ℕ), (∀ (i j : Fin n), c.val (i, j) < m) → m ≥ n) :=
sorry

end min_colors_regular_ngon_l29_2924


namespace cindy_envelopes_l29_2960

theorem cindy_envelopes (initial : ℕ) (friend1 friend2 friend3 friend4 friend5 : ℕ) :
  initial = 137 →
  friend1 = 4 →
  friend2 = 7 →
  friend3 = 5 →
  friend4 = 10 →
  friend5 = 3 →
  initial - (friend1 + friend2 + friend3 + friend4 + friend5) = 108 :=
by
  sorry

end cindy_envelopes_l29_2960


namespace sideEdgeLength_of_rightTriangularPyramid_l29_2932

/-- A right triangular pyramid with three mutually perpendicular side edges of equal length -/
structure RightTriangularPyramid where
  sideEdgeLength : ℝ
  mutuallyPerpendicular : Bool
  equalLength : Bool

/-- The circumscribed sphere of a RightTriangularPyramid -/
def circumscribedSphere (pyramid : RightTriangularPyramid) : ℝ → Prop :=
  fun surfaceArea => surfaceArea = 4 * Real.pi

/-- Theorem: The length of a side edge of a right triangular pyramid is 2√3/3 -/
theorem sideEdgeLength_of_rightTriangularPyramid (pyramid : RightTriangularPyramid) 
  (h1 : pyramid.mutuallyPerpendicular = true)
  (h2 : pyramid.equalLength = true)
  (h3 : circumscribedSphere pyramid (4 * Real.pi)) :
  pyramid.sideEdgeLength = 2 * Real.sqrt 3 / 3 := by
  sorry

end sideEdgeLength_of_rightTriangularPyramid_l29_2932


namespace sum_nonnegative_implies_one_nonnegative_l29_2927

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) : 
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end sum_nonnegative_implies_one_nonnegative_l29_2927


namespace twelfth_term_of_specific_arithmetic_sequence_l29_2957

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let a2 := (5 : ℚ) / 6
  let d := a2 - a
  arithmeticSequence a d 12 = (25 : ℚ) / 6 := by
  sorry

end twelfth_term_of_specific_arithmetic_sequence_l29_2957


namespace necessary_not_sufficient_condition_l29_2901

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x ≥ 1, monotonically_increasing (f a) 1 x) →
  (a ≤ -2 ∧ ∃ b, b ≤ 0 ∧ b > -2 ∧ ∀ x ≥ 1, monotonically_increasing (f b) 1 x) :=
sorry

end necessary_not_sufficient_condition_l29_2901


namespace largest_base_for_12_cubed_digit_sum_base_8_digit_sum_not_9_twelve_cubed_base_10_sum_of_digits_1728_base_10_largest_base_for_12_cubed_digit_sum_not_9_l29_2904

/-- The sum of digits of a natural number in a given base -/
def sum_of_digits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- The representation of a natural number in a given base -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem largest_base_for_12_cubed_digit_sum :
  ∀ b : ℕ, b > 8 → sum_of_digits (12^3) b = 3^2 := by sorry

theorem base_8_digit_sum_not_9 :
  sum_of_digits (12^3) 8 ≠ 3^2 := by sorry

theorem twelve_cubed_base_10 :
  12^3 = 1728 := by sorry

theorem sum_of_digits_1728_base_10 :
  sum_of_digits 1728 10 = 3^2 := by sorry

/-- 8 is the largest base b such that the sum of the base-b digits of 12^3 is not equal to 3^2 -/
theorem largest_base_for_12_cubed_digit_sum_not_9 :
  ∀ b : ℕ, b > 8 → sum_of_digits (12^3) b = 3^2 ∧
  sum_of_digits (12^3) 8 ≠ 3^2 := by sorry

end largest_base_for_12_cubed_digit_sum_base_8_digit_sum_not_9_twelve_cubed_base_10_sum_of_digits_1728_base_10_largest_base_for_12_cubed_digit_sum_not_9_l29_2904


namespace slope_tangent_ln_at_3_l29_2941

/-- The slope of the tangent line to y = ln x at x = 3 is 1/3 -/
theorem slope_tangent_ln_at_3 : 
  let f : ℝ → ℝ := λ x => Real.log x
  HasDerivAt f (1/3) 3 := by sorry

end slope_tangent_ln_at_3_l29_2941


namespace perfect_square_mod_three_l29_2914

theorem perfect_square_mod_three (n : ℤ) : (n^2) % 3 = 0 ∨ (n^2) % 3 = 1 := by
  sorry

end perfect_square_mod_three_l29_2914


namespace ratio_equality_product_l29_2992

theorem ratio_equality_product (x : ℝ) :
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) →
  ∃ y : ℝ, (2 * y + 3) / (3 * y + 3) = (5 * y + 4) / (8 * y + 4) ∧ y ≠ x ∧ x * y = 0 :=
by sorry

end ratio_equality_product_l29_2992


namespace unique_tiling_l29_2984

/-- A set is bounded below or above -/
def BoundedBelowOrAbove (A : Set ℝ) : Prop :=
  (∃ l, ∀ a ∈ A, l ≤ a) ∨ (∃ u, ∀ a ∈ A, a ≤ u)

/-- S tiles A -/
def Tiles (S A : Set ℝ) : Prop :=
  ∃ (I : Type) (f : I → Set ℝ), (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A

/-- Unique tiling -/
def UniqueTiling (S A : Set ℝ) : Prop :=
  ∀ (I J : Type) (f : I → Set ℝ) (g : J → Set ℝ),
    (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A →
    (∀ i, g i ⊆ S) ∧ (∀ i j, i ≠ j → g i ∩ g j = ∅) ∧ (⋃ i, g i) = A →
    ∃ (h : I ≃ J), ∀ i, f i = g (h i)

theorem unique_tiling (A : Set ℝ) (S : Set ℝ) :
  BoundedBelowOrAbove A → Tiles S A → UniqueTiling S A := by
  sorry

end unique_tiling_l29_2984


namespace veridux_female_employees_l29_2981

/-- Proves that the number of female employees at Veridux Corporation is 90 -/
theorem veridux_female_employees :
  let total_employees : ℕ := 250
  let total_managers : ℕ := 40
  let male_associates : ℕ := 160
  let female_managers : ℕ := 40
  let total_associates : ℕ := total_employees - total_managers
  let female_associates : ℕ := total_associates - male_associates
  let female_employees : ℕ := female_managers + female_associates
  female_employees = 90 := by
  sorry


end veridux_female_employees_l29_2981


namespace linear_function_properties_l29_2930

/-- A linear function of the form y = mx + 4m - 2 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m - 2

theorem linear_function_properties :
  ∃ m : ℝ, 
    (∃ y : ℝ, y ≠ -2 ∧ linear_function m 0 = y) ∧ 
    (let f := linear_function (1/3);
     ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ f x₁ > 0 ∧ x₂ < 0 ∧ f x₂ < 0 ∧ x₃ > 0 ∧ f x₃ < 0) ∧
    (linear_function (1/2) 0 = 0) ∧
    (∀ m : ℝ, linear_function m (-4) = -2) :=
by sorry

end linear_function_properties_l29_2930


namespace min_nSn_l29_2928

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
  S_10 : S 10 = 0
  S_15 : S 15 = 25

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) : 
  (∃ n : ℕ, n > 0 ∧ n * seq.S n = -49) ∧ 
  (∀ m : ℕ, m > 0 → m * seq.S m ≥ -49) := by
  sorry

end min_nSn_l29_2928


namespace factorization_of_9a_minus_6b_l29_2955

theorem factorization_of_9a_minus_6b (a b : ℝ) : 9*a - 6*b = 3*(3*a - 2*b) := by
  sorry

end factorization_of_9a_minus_6b_l29_2955


namespace fruit_shop_apples_l29_2970

/-- Given a ratio of fruits and the number of mangoes, calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples : ℕ) 
  (num_mangoes : ℕ) (h1 : ratio_mangoes = 10) (h2 : ratio_oranges = 2) 
  (h3 : ratio_apples = 3) (h4 : num_mangoes = 120) : 
  (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

end fruit_shop_apples_l29_2970


namespace new_plan_cost_theorem_l29_2905

def old_phone_plan_cost : ℝ := 150
def old_internet_cost : ℝ := 50
def old_calling_cost : ℝ := 30
def old_streaming_cost : ℝ := 40

def new_phone_plan_increase : ℝ := 0.30
def new_internet_increase : ℝ := 0.20
def new_calling_discount : ℝ := 0.15
def new_streaming_increase : ℝ := 0.25
def promotional_discount : ℝ := 0.10

def new_phone_plan_cost : ℝ := old_phone_plan_cost * (1 + new_phone_plan_increase)
def new_internet_cost : ℝ := old_internet_cost * (1 + new_internet_increase)
def new_calling_cost : ℝ := old_calling_cost * (1 - new_calling_discount)
def new_streaming_cost : ℝ := old_streaming_cost * (1 + new_streaming_increase)

def total_cost_before_discount : ℝ := 
  new_phone_plan_cost + new_internet_cost + new_calling_cost + new_streaming_cost

def total_cost_after_discount : ℝ := 
  total_cost_before_discount * (1 - promotional_discount)

theorem new_plan_cost_theorem : 
  total_cost_after_discount = 297.45 := by sorry

end new_plan_cost_theorem_l29_2905


namespace complex_modulus_l29_2910

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l29_2910


namespace f_is_odd_l29_2911

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0

-- The functional equation
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b

-- The theorem to prove
theorem f_is_odd : (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b) → 
  (∀ x : ℝ, f (-x) = -f x) :=
sorry

end f_is_odd_l29_2911


namespace max_value_of_expression_l29_2974

theorem max_value_of_expression (a b : ℝ) 
  (h : 17 * (a^2 + b^2) - 30 * a * b - 16 = 0) : 
  ∃ (x : ℝ), x = Real.sqrt (16 * a^2 + 4 * b^2 - 16 * a * b - 12 * a + 6 * b + 9) ∧ 
  x ≤ 7 ∧ 
  ∃ (a₀ b₀ : ℝ), 17 * (a₀^2 + b₀^2) - 30 * a₀ * b₀ - 16 = 0 ∧ 
    Real.sqrt (16 * a₀^2 + 4 * b₀^2 - 16 * a₀ * b₀ - 12 * a₀ + 6 * b₀ + 9) = 7 :=
by sorry

end max_value_of_expression_l29_2974


namespace random_sampling_cannot_prove_inequality_l29_2943

-- Define the type for inequality proof methods
inductive InequalityProofMethod
  | Comparison
  | Synthetic
  | Analytic
  | Contradiction
  | Scaling
  | RandomSampling

-- Define a predicate for methods that can prove inequalities
def can_prove_inequality (method : InequalityProofMethod) : Prop :=
  match method with
  | InequalityProofMethod.Comparison => True
  | InequalityProofMethod.Synthetic => True
  | InequalityProofMethod.Analytic => True
  | InequalityProofMethod.Contradiction => True
  | InequalityProofMethod.Scaling => True
  | InequalityProofMethod.RandomSampling => False

-- Define random sampling as a sampling method
def is_sampling_method (method : InequalityProofMethod) : Prop :=
  method = InequalityProofMethod.RandomSampling

-- Theorem stating that random sampling cannot be used to prove inequalities
theorem random_sampling_cannot_prove_inequality :
  ∀ (method : InequalityProofMethod),
    is_sampling_method method → ¬(can_prove_inequality method) :=
by sorry

end random_sampling_cannot_prove_inequality_l29_2943


namespace tulip_bouquet_combinations_l29_2954

theorem tulip_bouquet_combinations (n : ℕ) (max_tulips : ℕ) (total_money : ℕ) (tulip_cost : ℕ) : 
  n = 11 → 
  max_tulips = 11 → 
  total_money = 550 → 
  tulip_cost = 49 → 
  (Finset.filter (fun k => k % 2 = 1 ∧ k ≤ max_tulips) (Finset.range (n + 1))).card = 2^(n - 1) := by
  sorry

end tulip_bouquet_combinations_l29_2954


namespace goldbach_counterexample_characterization_l29_2915

-- Define Goldbach's conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define what constitutes a counterexample to Goldbach's conjecture
def is_goldbach_counterexample (n : ℕ) : Prop :=
  n > 2 ∧ Even n ∧ ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q)

-- The theorem to prove
theorem goldbach_counterexample_characterization :
  ∀ n : ℕ, is_goldbach_counterexample n ↔ ¬goldbach_conjecture :=
by sorry

end goldbach_counterexample_characterization_l29_2915


namespace eco_park_cherry_sample_l29_2975

/-- Represents the number of cherry trees in a stratified sample -/
def cherry_trees_in_sample (total_trees : ℕ) (total_cherry_trees : ℕ) (sample_size : ℕ) : ℕ :=
  (total_cherry_trees * sample_size) / total_trees

/-- Theorem stating the number of cherry trees in the sample for the given eco-park -/
theorem eco_park_cherry_sample :
  cherry_trees_in_sample 60000 4000 300 = 20 := by
  sorry

end eco_park_cherry_sample_l29_2975


namespace f_seven_equals_163_l29_2990

theorem f_seven_equals_163 (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + 8 * x * y - 2) : 
  f 7 = 163 := by
sorry

end f_seven_equals_163_l29_2990


namespace unique_solution_l29_2922

/-- Represents a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ → ℕ
| 0 => n % base
| k + 1 => baseRepresentation (n / base) base k

/-- The equation to be solved --/
def equationHolds (x : ℕ) : Prop :=
  baseRepresentation 2016 x 3 * x^3 +
  baseRepresentation 2016 x 2 * x^2 +
  baseRepresentation 2016 x 1 * x +
  baseRepresentation 2016 x 0 = x^3 + 2*x + 342

theorem unique_solution :
  ∃! x : ℕ, x > 0 ∧ equationHolds x ∧ x = 7 :=
sorry

end unique_solution_l29_2922


namespace units_digit_of_fraction_l29_2920

theorem units_digit_of_fraction : ∃ n : ℕ, n % 10 = 4 ∧ (30 * 31 * 32 * 33 * 34) / 400 = n := by
  sorry

end units_digit_of_fraction_l29_2920


namespace class_average_l29_2909

theorem class_average (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = 40 →
  group1_students = 28 →
  group2_students = 12 →
  group1_average = 68 / 100 →
  group2_average = 77 / 100 →
  let total_score := group1_students * group1_average + group2_students * group2_average
  let class_average := total_score / total_students
  class_average = 707 / 1000 := by
  sorry

end class_average_l29_2909


namespace min_value_of_expression_l29_2959

theorem min_value_of_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end min_value_of_expression_l29_2959


namespace bill_calculation_l29_2989

theorem bill_calculation (a b c : ℤ) 
  (h1 : a - (b - c) = 13)
  (h2 : (b - c) - a = -9)
  (h3 : a - b - c = 1) : 
  b - c = 1 := by
sorry

end bill_calculation_l29_2989


namespace race_result_l29_2977

/-- Represents the difference in meters between two runners at the end of a 1000-meter race. -/
def finish_difference (runner1 runner2 : ℕ) : ℝ := sorry

theorem race_result (A B C : ℕ) :
  finish_difference A C = 200 →
  finish_difference B C = 120.87912087912093 →
  finish_difference A B = 79.12087912087907 :=
by sorry

end race_result_l29_2977


namespace best_discount_l29_2952

def original_price : ℝ := 100

def discount_a (price : ℝ) : ℝ := price * 0.8

def discount_b (price : ℝ) : ℝ := price * 0.9 * 0.9

def discount_c (price : ℝ) : ℝ := price * 0.85 * 0.95

def discount_d (price : ℝ) : ℝ := price * 0.95 * 0.85

theorem best_discount :
  discount_a original_price < discount_b original_price ∧
  discount_a original_price < discount_c original_price ∧
  discount_a original_price < discount_d original_price :=
sorry

end best_discount_l29_2952


namespace difference_of_y_coordinates_is_two_l29_2908

noncomputable def e : ℝ := Real.exp 1

theorem difference_of_y_coordinates_is_two :
  ∀ a b : ℝ,
  (a^2 + e^4 = 2 * e^2 * a + 1) →
  (b^2 + e^4 = 2 * e^2 * b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end difference_of_y_coordinates_is_two_l29_2908


namespace survey_respondents_l29_2936

/-- Represents the number of people preferring each brand in a survey. -/
structure SurveyPreferences where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (prefs : SurveyPreferences) : ℕ :=
  prefs.x + prefs.y + prefs.z

/-- Theorem stating the total number of respondents in the survey. -/
theorem survey_respondents : ∃ (prefs : SurveyPreferences), 
  prefs.x = 360 ∧ 
  prefs.x * 4 = prefs.y * 9 ∧ 
  prefs.x * 3 = prefs.z * 9 ∧ 
  totalRespondents prefs = 640 := by
  sorry


end survey_respondents_l29_2936


namespace hexagon_ABCDEF_perimeter_l29_2900

def hexagon_perimeter (AB BC CD DE EF AF : ℝ) : ℝ :=
  AB + BC + CD + DE + EF + AF

theorem hexagon_ABCDEF_perimeter :
  ∀ (AB BC CD DE EF AF : ℝ),
    AB = 1 → BC = 1 → CD = 1 → DE = 1 → EF = 1 → AF = Real.sqrt 5 →
    hexagon_perimeter AB BC CD DE EF AF = 5 + Real.sqrt 5 := by
  sorry

end hexagon_ABCDEF_perimeter_l29_2900


namespace focus_coordinates_l29_2969

/-- A parabola with equation x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- The directrix of a parabola -/
def directrix (par : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y = -2

/-- The focus of a parabola -/
def focus (par : Parabola) : ℝ × ℝ :=
  (0, par.p)

/-- Theorem stating that if the directrix of a parabola passes through (0, -2),
    then its focus is at (0, 2) -/
theorem focus_coordinates (par : Parabola) :
  directrix par 0 (-2) → focus par = (0, 2) := by
  sorry

end focus_coordinates_l29_2969


namespace tetrahedron_probabilities_l29_2951

/-- A regular tetrahedron with numbers 1, 2, 3, 4 on its faces -/
structure Tetrahedron :=
  (faces : Fin 4 → Fin 4)
  (bijective : Function.Bijective faces)

/-- The probability space of throwing the tetrahedron twice -/
def TetrahedronThrows := Tetrahedron × Tetrahedron

/-- Event A: 2 or 3 facing down on first throw -/
def event_A (t : TetrahedronThrows) : Prop :=
  t.1.faces 0 = 2 ∨ t.1.faces 0 = 3

/-- Event B: sum of numbers facing down is odd -/
def event_B (t : TetrahedronThrows) : Prop :=
  (t.1.faces 0 + t.2.faces 0) % 2 = 1

/-- Event C: sum of numbers facing up is not less than 15 -/
def event_C (t : TetrahedronThrows) : Prop :=
  (10 - t.1.faces 0 - t.2.faces 0) ≥ 15

/-- The probability measure on TetrahedronThrows -/
noncomputable def P : Set TetrahedronThrows → ℝ := sorry

theorem tetrahedron_probabilities :
  (P {t : TetrahedronThrows | event_A t} * P {t : TetrahedronThrows | event_B t} =
   P {t : TetrahedronThrows | event_A t ∧ event_B t}) ∧
  (P {t : TetrahedronThrows | event_A t ∨ event_B t} = 3/4) ∧
  (P {t : TetrahedronThrows | event_C t} = 5/8) :=
sorry

end tetrahedron_probabilities_l29_2951


namespace sphere_inscribed_in_all_shapes_l29_2980

/-- Represents a sphere with a given diameter -/
structure Sphere where
  diameter : ℝ

/-- Represents a square-based prism with a given base edge -/
structure SquarePrism where
  baseEdge : ℝ

/-- Represents a triangular prism with an isosceles triangle base -/
structure TriangularPrism where
  base : ℝ
  height : ℝ

/-- Represents a cylinder with a given base circle diameter -/
structure Cylinder where
  baseDiameter : ℝ

/-- Predicate to check if a sphere can be inscribed in a square prism -/
def inscribedInSquarePrism (s : Sphere) (p : SquarePrism) : Prop :=
  s.diameter = p.baseEdge

/-- Predicate to check if a sphere can be inscribed in a triangular prism -/
def inscribedInTriangularPrism (s : Sphere) (p : TriangularPrism) : Prop :=
  s.diameter = p.base ∧ s.diameter ≤ p.height * (Real.sqrt 3) / 2

/-- Predicate to check if a sphere can be inscribed in a cylinder -/
def inscribedInCylinder (s : Sphere) (c : Cylinder) : Prop :=
  s.diameter = c.baseDiameter

/-- Theorem stating that a sphere with diameter a can be inscribed in all three shapes -/
theorem sphere_inscribed_in_all_shapes (a : ℝ) :
  let s : Sphere := ⟨a⟩
  let sp : SquarePrism := ⟨a⟩
  let tp : TriangularPrism := ⟨a, a⟩
  let c : Cylinder := ⟨a⟩
  inscribedInSquarePrism s sp ∧
  inscribedInTriangularPrism s tp ∧
  inscribedInCylinder s c :=
by sorry

end sphere_inscribed_in_all_shapes_l29_2980


namespace rhombus_longer_diagonal_l29_2986

/-- Given a rhombus with side length 51 and shorter diagonal 48, prove that its longer diagonal is 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 51 → shorter_diagonal = 48 → longer_diagonal = 90 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end rhombus_longer_diagonal_l29_2986


namespace student_a_score_l29_2987

/-- Calculates the score for a test based on the given grading method -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

theorem student_a_score :
  calculateScore 100 90 = 70 := by
  sorry

end student_a_score_l29_2987


namespace polar_to_rectangular_equation_l29_2973

/-- The rectangular coordinate equation of the curve ρ = sin θ - 3cos θ -/
theorem polar_to_rectangular_equation :
  ∀ (x y ρ θ : ℝ),
  (ρ = Real.sin θ - 3 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 - 3*x + y^2 - y = 0) := by
  sorry

end polar_to_rectangular_equation_l29_2973


namespace x_minus_y_positive_l29_2979

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by
sorry

end x_minus_y_positive_l29_2979


namespace factorization_of_polynomial_l29_2971

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end factorization_of_polynomial_l29_2971


namespace tommy_books_l29_2978

/-- The number of books Tommy wants to buy -/
def num_books (book_cost savings_needed current_money : ℕ) : ℕ :=
  (savings_needed + current_money) / book_cost

/-- Proof that Tommy wants to buy 8 books -/
theorem tommy_books : num_books 5 27 13 = 8 := by
  sorry

end tommy_books_l29_2978


namespace upper_limit_range_l29_2916

theorem upper_limit_range (n x : ℝ) : 
  3 < n ∧ n < x ∧ 6 < n ∧ n < 10 ∧ n = 7 → x > 7 := by
sorry

end upper_limit_range_l29_2916


namespace complex_equation_solution_l29_2902

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I)
  (h2 : Complex.abs z = Real.sqrt 2)
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end complex_equation_solution_l29_2902


namespace base5_division_theorem_l29_2926

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Represents a number in base 5 -/
structure Base5 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

theorem base5_division_theorem (a b : Base5) :
  let a_base10 := base5ToBase10 a.digits
  let b_base10 := base5ToBase10 b.digits
  let quotient_base10 := a_base10 / b_base10
  let quotient_base5 := Base5.mk (base10ToBase5 quotient_base10) sorry
  a = Base5.mk [1, 3, 2, 4] sorry ∧ 
  b = Base5.mk [1, 2] sorry → 
  quotient_base5 = Base5.mk [1, 1, 0] sorry := by
  sorry

end base5_division_theorem_l29_2926


namespace triangle_similarity_l29_2942

/-- Given five complex numbers a, b, c, u, v representing points on a plane,
    if the ratios (v-a)/(u-a), (u-v)/(b-v), and (c-u)/(v-u) are equal,
    then the ratio (v-a)/(u-a) is equal to (c-a)/(b-a). -/
theorem triangle_similarity (a b c u v : ℂ) :
  (v - a) / (u - a) = (u - v) / (b - v) ∧
  (v - a) / (u - a) = (c - u) / (v - u) →
  (v - a) / (u - a) = (c - a) / (b - a) :=
by sorry

end triangle_similarity_l29_2942


namespace power_product_equality_l29_2940

theorem power_product_equality (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end power_product_equality_l29_2940


namespace last_bead_color_l29_2931

def bead_colors := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def necklace_length : Nat := 85

theorem last_bead_color (h : necklace_length = 85) :
  bead_colors[(necklace_length - 1) % bead_colors.length] = "yellow" := by
  sorry

end last_bead_color_l29_2931


namespace intersection_of_M_and_N_l29_2945

open Set

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x > 1}
  let N : Set ℝ := {x | x^2 - 2*x < 0}
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l29_2945


namespace modulus_of_complex_power_l29_2999

theorem modulus_of_complex_power :
  Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end modulus_of_complex_power_l29_2999


namespace cricket_team_age_theorem_l29_2918

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) (bowlers_count : ℕ) 
  (bowlers_min_age : ℕ) (bowlers_max_age : ℕ) : Prop :=
  let wicket_keeper_age := captain_age + wicket_keeper_age_diff
  let total_age := team_size * 30
  let captain_wicket_keeper_age := captain_age + wicket_keeper_age
  let remaining_players := team_size - 2
  total_age = captain_wicket_keeper_age + remaining_players * (30 - remaining_players_age_diff) ∧
  bowlers_min_age * bowlers_count ≤ bowlers_count * 30 ∧
  bowlers_count * 30 ≤ bowlers_max_age * bowlers_count

theorem cricket_team_age_theorem : 
  cricket_team_age_problem 11 24 3 1 5 18 22 := by
  sorry

end cricket_team_age_theorem_l29_2918


namespace concentric_circles_area_ratio_l29_2944

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ := d₁ / 2  -- radius of smallest circle
  let r₂ := d₂ / 2  -- radius of middle circle
  let r₃ := d₃ / 2  -- radius of largest circle
  let A₁ := π * r₁^2  -- area of smallest circle
  let A₂ := π * r₂^2  -- area of middle circle
  let A₃ := π * r₃^2  -- area of largest circle
  let blue_area := A₂ - A₁  -- area between smallest and middle circles
  let green_area := A₃ - A₂  -- area between middle and largest circles
  (green_area / blue_area : ℝ) = 5/3
  := by sorry

end concentric_circles_area_ratio_l29_2944


namespace modulus_of_z_l29_2947

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + Real.sqrt 3 * i) * z = 4

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l29_2947


namespace subset_sum_divisible_by_2n_l29_2964

theorem subset_sum_divisible_by_2n
  (n : ℕ)
  (h_n : n ≥ 4)
  (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bounds : ∀ i : Fin n, 0 < a i ∧ a i < 2*n) :
  ∃ (S : Finset (Fin n)), (S.sum (λ i => a i)) % (2*n) = 0 :=
sorry

end subset_sum_divisible_by_2n_l29_2964


namespace mountain_hike_l29_2985

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 3 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 9 := by
  sorry

end mountain_hike_l29_2985


namespace square_is_self_product_l29_2976

theorem square_is_self_product (b : ℚ) : b^2 = b * b := by
  sorry

end square_is_self_product_l29_2976


namespace average_speed_to_destination_l29_2919

/-- Proves that given a round trip with a total one-way distance of 150 km,
    a return speed of 30 km/hr, and an average speed for the whole journey of 37.5 km/hr,
    the average speed while traveling to the place is 50 km/hr. -/
theorem average_speed_to_destination (v : ℝ) : 
  (150 : ℝ) / v + (150 : ℝ) / 30 = 300 / (37.5 : ℝ) → v = 50 := by
  sorry

end average_speed_to_destination_l29_2919


namespace equation_real_roots_a_range_l29_2938

theorem equation_real_roots_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 2 - 2^(-|x-2|) = 2 + a) → -1 ≤ a ∧ a < 0 :=
by sorry

end equation_real_roots_a_range_l29_2938


namespace elizabeth_stickers_l29_2917

/-- Calculates the total number of stickers used on water bottles -/
def total_stickers (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (stickers_per_bottle : ℕ) : ℕ :=
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle

/-- Theorem: Given Elizabeth's specific situation, she uses 21 stickers in total -/
theorem elizabeth_stickers : 
  total_stickers 10 2 1 3 = 21 := by
  sorry

end elizabeth_stickers_l29_2917


namespace hyperbola_intersection_theorem_l29_2925

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the focal points
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem hyperbola_intersection_theorem 
  (k m : ℝ) 
  (A B : ℝ × ℝ) 
  (h_focal_length : Real.sqrt 16 = 4)
  (h_imaginary_axis : Real.sqrt 4 = 2)
  (h_m_nonzero : m ≠ 0)
  (h_distinct : A ≠ B)
  (h_on_hyperbola_A : hyperbola A.1 A.2)
  (h_on_hyperbola_B : hyperbola B.1 B.2)
  (h_on_line_A : line k m A.1 A.2)
  (h_on_line_B : line k m B.1 B.2)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3)
  (h_passes_F₂ : line k m F₂.1 F₂.2) :
  -- 1. Eccentricity
  (2 * Real.sqrt 3 / 3 = Real.sqrt (1 - 1 / 3)) ∧
  -- 2. Equation of line l
  ((k = 1 ∧ m = -2) ∨ (k = -1 ∧ m = 2) ∨ (k = 0 ∧ m ≠ 0)) ∧
  -- 3. Range of m
  (k ≠ 0 → m ∈ Set.Icc (-1/4) 0 ∪ Set.Ioi 4) ∧
  (k = 0 → m ∈ Set.univ \ {0}) := by
  sorry

end hyperbola_intersection_theorem_l29_2925


namespace rectangle_area_perimeter_relation_l29_2997

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 8
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end rectangle_area_perimeter_relation_l29_2997


namespace custom_mul_ab_equals_nine_l29_2991

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - 1

/-- Theorem stating that under given conditions, a*b = 9 -/
theorem custom_mul_ab_equals_nine
  (a b : ℝ)
  (h1 : custom_mul a b 1 2 = 4)
  (h2 : custom_mul a b (-2) 3 = 10) :
  custom_mul a b a b = 9 :=
sorry

end custom_mul_ab_equals_nine_l29_2991


namespace fraction_equality_l29_2961

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 2) :
  (a - b) / a = 3 / 5 := by
  sorry

end fraction_equality_l29_2961


namespace sum_of_even_coefficients_l29_2972

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x : ℤ, (2*x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₂ + a₄ + a₆ = 364 := by
sorry

end sum_of_even_coefficients_l29_2972


namespace equation_solutions_l29_2935

theorem equation_solutions :
  (∀ x : ℚ, (1/2 * x - 2 = 4 + 1/3 * x) ↔ (x = 36)) ∧
  (∀ x : ℚ, ((x - 1) / 4 - 2 = (2 * x - 3) / 6) ↔ (x = -21)) ∧
  (∀ x : ℚ, (1/3 * (x - 1/2 * (x - 1)) = 2/3 * (x - 1/2)) ↔ (x = 1)) ∧
  (∀ x : ℚ, (x / (7/10) - (17/100 - 1/5 * x) / (3/100) = 1) ↔ (x = 14/17)) :=
by sorry

end equation_solutions_l29_2935


namespace smallest_n_congruence_l29_2903

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 1846 [ZMOD 26] → n ≤ m) ∧ 
  (5 * n ≡ 1846 [ZMOD 26]) :=
by sorry

end smallest_n_congruence_l29_2903


namespace geometry_algebra_properties_l29_2996

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define congruent angles
def congruent (α β : Real) : Prop := α = β

-- Define vertical angles
def vertical (α β : Real) : Prop := α = β

-- Define perpendicular lines
def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem geometry_algebra_properties :
  (∃ α β : Real, supplementary α β ∧ ¬congruent α β) ∧
  (∀ α β : Real, vertical α β → α = β) ∧
  ((-1 : Real)^(1/3) = -1) ∧
  (∀ l₁ l₂ l₃ : Line, perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel l₁ l₂) :=
sorry

end geometry_algebra_properties_l29_2996


namespace doll_problem_l29_2958

theorem doll_problem (S : ℕ+) (D : ℕ) 
  (h1 : 4 * S + 3 = D) 
  (h2 : 5 * S = D + 6) : 
  D = 39 := by
sorry

end doll_problem_l29_2958


namespace ratio_a_to_c_l29_2950

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end ratio_a_to_c_l29_2950


namespace rectangle_to_squares_l29_2907

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Function to cut a rectangle in half across its length -/
def cutRectangleInHalf (r : Rectangle) : Square :=
  { side := r.width }

theorem rectangle_to_squares (r : Rectangle) 
  (h1 : r.length = 10)
  (h2 : r.width = 5) :
  (cutRectangleInHalf r).side = 5 := by
  sorry

end rectangle_to_squares_l29_2907


namespace dads_age_l29_2913

theorem dads_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 9 →
  age_difference = 27 →
  (4 : ℕ) * son_age + age_difference = 63 := by
sorry

end dads_age_l29_2913


namespace faculty_size_l29_2939

/-- The number of second year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The percentage of second year students in the total student body -/
def second_year_percentage : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 674

theorem faculty_size : 
  ∃ (second_year_students : ℕ), 
    second_year_students = numeric_methods + automatic_control - both_subjects ∧
    (second_year_students : ℚ) / total_students = second_year_percentage :=
by sorry

end faculty_size_l29_2939


namespace expected_value_is_500_l29_2983

/-- Represents the prize structure for a game activity -/
structure PrizeStructure where
  firstPrize : ℝ
  commonDifference : ℝ

/-- Represents the probability distribution for winning prizes -/
structure ProbabilityDistribution where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Calculates the expected value of the prize -/
def expectedValue (ps : PrizeStructure) (pd : ProbabilityDistribution) : ℝ :=
  let secondPrize := ps.firstPrize + ps.commonDifference
  let thirdPrize := ps.firstPrize + 2 * ps.commonDifference
  let secondProb := pd.firstTerm * pd.commonRatio
  let thirdProb := pd.firstTerm * pd.commonRatio * pd.commonRatio
  ps.firstPrize * pd.firstTerm + secondPrize * secondProb + thirdPrize * thirdProb

/-- The main theorem stating that the expected value is 500 yuan -/
theorem expected_value_is_500 
  (ps : PrizeStructure) 
  (pd : ProbabilityDistribution) 
  (h1 : ps.firstPrize = 700)
  (h2 : ps.commonDifference = -140)
  (h3 : pd.commonRatio = 2)
  (h4 : pd.firstTerm + pd.firstTerm * pd.commonRatio + pd.firstTerm * pd.commonRatio * pd.commonRatio = 1) :
  expectedValue ps pd = 500 := by
  sorry

end expected_value_is_500_l29_2983


namespace parabola_vertex_vertex_of_specific_parabola_l29_2982

/-- The vertex of a parabola y = ax^2 + bx + c is (h, k) where h = -b/(2a) and k = f(h) -/
theorem parabola_vertex (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let vertex_y : ℝ := f vertex_x
  (∀ x, f x ≥ vertex_y) ∨ (∀ x, f x ≤ vertex_y) :=
sorry

theorem vertex_of_specific_parabola :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 5
  let vertex : ℝ × ℝ := (1, 2)
  (∀ x, f x ≥ 2) ∧ f 1 = 2 :=
sorry

end parabola_vertex_vertex_of_specific_parabola_l29_2982


namespace cubic_factorization_l29_2949

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m + 2)*(m - 2) := by
  sorry

end cubic_factorization_l29_2949


namespace largest_whole_number_less_than_100_over_7_l29_2946

theorem largest_whole_number_less_than_100_over_7 : 
  ∀ x : ℕ, x ≤ 14 ↔ 7 * x < 100 :=
by sorry

end largest_whole_number_less_than_100_over_7_l29_2946


namespace find_M_l29_2937

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2024 + 2025 + 2026) / M ∧ M = 552 := by
  sorry

end find_M_l29_2937


namespace original_price_is_20_l29_2929

/-- Represents the ticket pricing scenario for a concert --/
structure ConcertTickets where
  original_price : ℝ
  total_revenue : ℝ
  total_tickets : ℕ
  discount_40_count : ℕ
  discount_15_count : ℕ

/-- The concert ticket scenario satisfies the given conditions --/
def valid_scenario (c : ConcertTickets) : Prop :=
  c.total_tickets = 50 ∧
  c.discount_40_count = 10 ∧
  c.discount_15_count = 20 ∧
  c.total_revenue = 860 ∧
  c.total_revenue = (c.discount_40_count : ℝ) * (0.6 * c.original_price) +
                    (c.discount_15_count : ℝ) * (0.85 * c.original_price) +
                    ((c.total_tickets - c.discount_40_count - c.discount_15_count) : ℝ) * c.original_price

/-- The original ticket price is $20 --/
theorem original_price_is_20 (c : ConcertTickets) (h : valid_scenario c) :
  c.original_price = 20 := by
  sorry

end original_price_is_20_l29_2929


namespace lcm_gcd_problem_l29_2995

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 55 →
  a = 210 →
  b = 605 := by
sorry

end lcm_gcd_problem_l29_2995


namespace games_purchase_l29_2912

theorem games_purchase (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 42 → spent_amount = 10 → game_cost = 8 → 
  (initial_amount - spent_amount) / game_cost = 4 := by
  sorry

end games_purchase_l29_2912


namespace incompatible_food_probability_l29_2933

-- Define the set of foods
def Food : Type := Fin 5

-- Define the incompatibility relation
def incompatible : Food → Food → Prop := sorry

-- Define the number of incompatible pairs
def num_incompatible_pairs : ℕ := 3

-- Define the total number of possible pairs
def total_pairs : ℕ := Nat.choose 5 2

-- State the theorem
theorem incompatible_food_probability :
  (num_incompatible_pairs : ℚ) / (total_pairs : ℚ) = 3 / 10 := by sorry

end incompatible_food_probability_l29_2933


namespace existence_of_pairs_l29_2906

theorem existence_of_pairs : ∃ (f : Fin 2018 → ℕ × ℕ),
  (∀ i : Fin 2018, (f i).1 ≠ (f i).2) ∧
  (∀ i : Fin 2017, (f i.succ).1 = (f i).1 + 1) ∧
  (∀ i : Fin 2017, (f i.succ).2 = (f i).2 + 1) ∧
  (∀ i : Fin 2018, (f i).1 % (f i).2 = 0) :=
by
  sorry

end existence_of_pairs_l29_2906


namespace expression_evaluation_l29_2965

theorem expression_evaluation :
  let x : ℚ := -2
  (3 + x * (3 + x) - 3^2) / (x - 3 + x^2) = 8 := by
sorry

end expression_evaluation_l29_2965


namespace line_equation_l29_2994

/-- A line passing through (1,1) and intersecting the circle (x-2)^2 + (y-3)^2 = 9 at two points A and B -/
def Line : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (k : ℝ), p.2 = k * (p.1 - 1) + 1}

/-- The circle (x-2)^2 + (y-3)^2 = 9 -/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 9}

/-- The line passes through (1,1) -/
axiom line_passes_through : (1, 1) ∈ Line

/-- The line intersects the circle at two points A and B -/
axiom line_intersects_circle : ∃ (A B : ℝ × ℝ), A ∈ Line ∩ Circle ∧ B ∈ Line ∩ Circle ∧ A ≠ B

/-- The distance between A and B is 4 -/
axiom distance_AB : ∀ (A B : ℝ × ℝ), A ∈ Line ∩ Circle → B ∈ Line ∩ Circle → A ≠ B →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

/-- The equation of the line is x + 2y - 3 = 0 -/
theorem line_equation : Line = {p : ℝ × ℝ | p.1 + 2 * p.2 - 3 = 0} :=
sorry

end line_equation_l29_2994


namespace students_in_class_l29_2998

def total_pencils : ℕ := 125
def pencils_per_student : ℕ := 5

theorem students_in_class : total_pencils / pencils_per_student = 25 := by
  sorry

end students_in_class_l29_2998


namespace child_ticket_cost_l29_2966

theorem child_ticket_cost
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 12)
  (h2 : total_tickets = 130)
  (h3 : total_receipts = 840)
  (h4 : adult_tickets = 40)
  : ∃ (child_price : ℕ),
    child_price = 4 ∧
    adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = total_receipts :=
by
  sorry

end child_ticket_cost_l29_2966


namespace number_puzzle_l29_2948

theorem number_puzzle :
  ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end number_puzzle_l29_2948


namespace f_is_linear_l29_2993

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation -x - 3 = 4 -/
def f (x : ℝ) : ℝ := -x - 3

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry


end f_is_linear_l29_2993
