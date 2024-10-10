import Mathlib

namespace linear_function_solution_l4037_403759

/-- A linear function passing through (0,2) with negative slope -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem linear_function_solution :
  ∀ k : ℝ, k < 0 → linearFunction (-1) = linearFunction k := by sorry

end linear_function_solution_l4037_403759


namespace arrangement_theorem_l4037_403710

/-- The number of ways to arrange 2 teachers and 5 students in a row,
    with the teachers adjacent but not at the ends. -/
def arrangement_count : ℕ := 960

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects,
    where order matters. -/
def permutations_of_k (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem arrangement_theorem :
  arrangement_count = 
    2 * permutations_of_k 5 2 * permutations 4 :=
by sorry

end arrangement_theorem_l4037_403710


namespace geometric_sequence_common_ratio_l4037_403792

/-- A geometric sequence with positive terms and a specific condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (2 * a 1 + a 2 = a 3)

/-- The common ratio of the geometric sequence is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h : GeometricSequence a) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n := by
  sorry


end geometric_sequence_common_ratio_l4037_403792


namespace arrangement_counts_l4037_403799

/-- The number of ways to arrange 3 boys and 4 girls in a row under specific conditions -/
theorem arrangement_counts :
  let total_people : ℕ := 7
  let num_boys : ℕ := 3
  let num_girls : ℕ := 4
  -- (1) Person A is neither at the middle nor at the ends
  (number_of_arrangements_1 : ℕ := 2880) →
  -- (2) Persons A and B must be at the two ends
  (number_of_arrangements_2 : ℕ := 240) →
  -- (3) Boys and girls alternate
  (number_of_arrangements_3 : ℕ := 144) →
  -- Prove all three conditions are true
  (number_of_arrangements_1 = 2880 ∧
   number_of_arrangements_2 = 240 ∧
   number_of_arrangements_3 = 144) :=
by sorry

end arrangement_counts_l4037_403799


namespace average_monthly_balance_l4037_403718

def monthly_balances : List ℝ := [200, 300, 250, 350, 300]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 280 := by
  sorry

end average_monthly_balance_l4037_403718


namespace scooter_initial_value_l4037_403755

/-- 
Given a scooter whose value depreciates to 3/4 of its value each year, 
if its value after one year is 30000, then its initial value was 40000.
-/
theorem scooter_initial_value (initial_value : ℝ) : 
  (3 / 4 : ℝ) * initial_value = 30000 → initial_value = 40000 := by
  sorry

end scooter_initial_value_l4037_403755


namespace investment_growth_l4037_403714

theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_amount = 705.03 →
  initial_investment * (1 + interest_rate) ^ years = final_amount :=
by sorry

end investment_growth_l4037_403714


namespace scientific_notation_of_1_097_billion_l4037_403716

theorem scientific_notation_of_1_097_billion :
  ∃ (a : ℝ) (n : ℤ), 1.097e9 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.097 ∧ n = 9 := by
  sorry

end scientific_notation_of_1_097_billion_l4037_403716


namespace geometric_sequence_min_a3_l4037_403777

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ q : ℝ, q > 0 → a 3 ≤ (a 1) * q^2) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end geometric_sequence_min_a3_l4037_403777


namespace prob_same_color_is_89_169_l4037_403797

def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5
def total_balls : ℕ := blue_balls + yellow_balls

def prob_same_color : ℚ := (blue_balls^2 + yellow_balls^2) / total_balls^2

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end prob_same_color_is_89_169_l4037_403797


namespace complement_union_theorem_l4037_403702

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {1, 3, 4, 5, 6} := by sorry

end complement_union_theorem_l4037_403702


namespace cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l4037_403747

/-- The surface area of a cube with volume equal to a 10x10x8 inch rectangular prism is 1200 square inches. -/
theorem cube_surface_area_from_prism_volume : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun prism_length prism_width prism_height cube_surface_area =>
    prism_length = 10 ∧
    prism_width = 10 ∧
    prism_height = 8 ∧
    cube_surface_area = 6 * (prism_length * prism_width * prism_height) ^ (2/3) ∧
    cube_surface_area = 1200

/-- Proof of the theorem -/
theorem cube_surface_area_from_prism_volume_proof :
  cube_surface_area_from_prism_volume 10 10 8 1200 := by
  sorry

#check cube_surface_area_from_prism_volume
#check cube_surface_area_from_prism_volume_proof

end cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l4037_403747


namespace all_ingredients_good_probability_l4037_403719

/-- The probability of selecting a fresh bottle of milk -/
def prob_fresh_milk : ℝ := 0.8

/-- The probability of selecting a good egg -/
def prob_good_egg : ℝ := 0.4

/-- The probability of selecting a good canister of flour -/
def prob_good_flour : ℝ := 0.75

/-- The probability that all three ingredients (milk, egg, flour) are good when selected randomly -/
def prob_all_good : ℝ := prob_fresh_milk * prob_good_egg * prob_good_flour

theorem all_ingredients_good_probability :
  prob_all_good = 0.18 := by
  sorry

end all_ingredients_good_probability_l4037_403719


namespace arithmetic_square_root_of_sqrt_16_l4037_403708

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l4037_403708


namespace gcf_of_180_and_270_l4037_403727

theorem gcf_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcf_of_180_and_270_l4037_403727


namespace square_root_three_expansion_l4037_403723

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  a = c ^ 2 + 3 * d ^ 2 ∧ b = 2 * c * d :=
sorry

end square_root_three_expansion_l4037_403723


namespace tomatoes_count_l4037_403775

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The difference between the number of students who suggested mashed potatoes
    and the number of students who suggested tomatoes -/
def difference : ℕ := 65

/-- The number of students who suggested adding tomatoes -/
def tomatoes : ℕ := mashed_potatoes - difference

theorem tomatoes_count : tomatoes = 79 := by
  sorry

end tomatoes_count_l4037_403775


namespace odd_square_minus_one_div_eight_l4037_403771

theorem odd_square_minus_one_div_eight (a : ℤ) (h : ∃ k : ℤ, a = 2 * k + 1) :
  ∃ m : ℤ, a^2 - 1 = 8 * m :=
by sorry

end odd_square_minus_one_div_eight_l4037_403771


namespace marksman_probability_l4037_403704

theorem marksman_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.20)
  (h2 : p9 = 0.30)
  (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end marksman_probability_l4037_403704


namespace tony_temp_day5_l4037_403752

-- Define the illnesses and their effects
structure Illness where
  duration : ℕ
  tempChange : ℤ
  startDay : ℕ

-- Define Tony's normal temperature and fever threshold
def normalTemp : ℕ := 95
def feverThreshold : ℕ := 100

-- Define the illnesses
def illnessA : Illness := ⟨7, 10, 1⟩
def illnessB : Illness := ⟨5, 4, 3⟩
def illnessC : Illness := ⟨3, -2, 5⟩

-- Function to calculate temperature change on a given day
def tempChangeOnDay (day : ℕ) : ℤ :=
  let baseChange := 
    (if day ≥ illnessA.startDay then illnessA.tempChange else 0) +
    (if day ≥ illnessB.startDay then 
      (if day ≥ illnessA.startDay then 2 * illnessB.tempChange else illnessB.tempChange)
    else 0) +
    (if day ≥ illnessC.startDay then illnessC.tempChange else 0)
  let synergisticEffect := if day = 5 then -3 else 0
  baseChange + synergisticEffect

-- Theorem to prove
theorem tony_temp_day5 : 
  (normalTemp : ℤ) + tempChangeOnDay 5 = 108 ∧ 
  (normalTemp : ℤ) + tempChangeOnDay 5 - feverThreshold = 8 := by
  sorry

end tony_temp_day5_l4037_403752


namespace largest_constant_inequality_l4037_403715

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D * (x + y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D' * (x + y)) → D' ≤ D) :=
by sorry

end largest_constant_inequality_l4037_403715


namespace point_set_is_hyperbola_l4037_403787

-- Define the set of points (x, y) based on the given parametric equations
def point_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≠ 0 ∧ p.1 = (2 * t + 1) / t ∧ p.2 = (t - 2) / t}

-- Theorem stating that the point_set forms a hyperbola
theorem point_set_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ 
    (∀ p : ℝ × ℝ, p ∈ point_set ↔ 
      a * p.1 * p.1 + b * p.1 * p.2 + c * p.2 * p.2 + d * p.1 + e * p.2 + f = 0) ∧
    b * b - 4 * a * c > 0 := by
  sorry

end point_set_is_hyperbola_l4037_403787


namespace exam_results_l4037_403758

/-- Given an examination where:
  * 35% of students failed in Hindi
  * 20% of students failed in both Hindi and English
  * 40% of students passed in both subjects
  Prove that 45% of students failed in English -/
theorem exam_results (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h_total : total = 100)
  (h_failed_hindi : failed_hindi = 35)
  (h_failed_both : failed_both = 20)
  (h_passed_both : passed_both = 40) :
  ∃ (failed_english : ℝ), failed_english = 45 :=
sorry

end exam_results_l4037_403758


namespace brown_sugar_amount_l4037_403756

-- Define the amount of white sugar used
def white_sugar : ℝ := 0.25

-- Define the additional amount of brown sugar compared to white sugar
def additional_brown_sugar : ℝ := 0.38

-- Theorem stating the amount of brown sugar used
theorem brown_sugar_amount : 
  white_sugar + additional_brown_sugar = 0.63 := by
  sorry

end brown_sugar_amount_l4037_403756


namespace outermost_to_innermost_ratio_l4037_403717

/-- A sequence of alternating inscribed squares and circles -/
structure SquareCircleSequence where
  S1 : Real  -- Side length of innermost square
  C1 : Real  -- Diameter of circle inscribing S1
  S2 : Real  -- Side length of square inscribing C1
  C2 : Real  -- Diameter of circle inscribing S2
  S3 : Real  -- Side length of square inscribing C2
  C3 : Real  -- Diameter of circle inscribing S3
  S4 : Real  -- Side length of outermost square

/-- Properties of the SquareCircleSequence -/
axiom sequence_properties (seq : SquareCircleSequence) :
  seq.C1 = seq.S1 * Real.sqrt 2 ∧
  seq.S2 = seq.C1 ∧
  seq.C2 = seq.S2 * Real.sqrt 2 ∧
  seq.S3 = seq.C2 ∧
  seq.C3 = seq.S3 * Real.sqrt 2 ∧
  seq.S4 = seq.C3

/-- The ratio of the outermost square's side length to the innermost square's side length is 2√2 -/
theorem outermost_to_innermost_ratio (seq : SquareCircleSequence) :
  seq.S4 / seq.S1 = 2 * Real.sqrt 2 := by
  sorry


end outermost_to_innermost_ratio_l4037_403717


namespace A_intersect_B_l4037_403798

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 1, 3} := by sorry

end A_intersect_B_l4037_403798


namespace proportion_problem_l4037_403738

theorem proportion_problem :
  ∀ x₁ x₂ x₃ x₄ : ℤ,
    (x₁ : ℚ) / x₂ = (x₃ : ℚ) / x₄ ∧
    x₁ = x₂ + 6 ∧
    x₃ = x₄ + 5 ∧
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = 793 →
    ((x₁ = -12 ∧ x₂ = -18 ∧ x₃ = -10 ∧ x₄ = -15) ∨
     (x₁ = 18 ∧ x₂ = 12 ∧ x₃ = 15 ∧ x₄ = 10)) :=
by sorry

end proportion_problem_l4037_403738


namespace element_correspondence_l4037_403712

-- Define the mapping f from A to B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem element_correspondence : f 2 = 5 := by
  sorry

end element_correspondence_l4037_403712


namespace merchant_profit_l4037_403776

theorem merchant_profit (C S : ℝ) (h : C > 0) (h1 : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end merchant_profit_l4037_403776


namespace sarahs_pool_depth_is_five_l4037_403729

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating that Sarah's pool depth is 5 feet -/
theorem sarahs_pool_depth_is_five :
  sarahs_pool_depth = 5 ∧
  johns_pool_depth = 2 * sarahs_pool_depth + 5 :=
by sorry

end sarahs_pool_depth_is_five_l4037_403729


namespace special_function_properties_l4037_403754

/-- A function satisfying the given functional equation -/
structure SpecialFunction where
  f : ℝ → ℝ
  eq : ∀ x y, f (x + y) * f (x - y) = f x + f y
  nonzero : f 0 ≠ 0

/-- Properties of the special function -/
theorem special_function_properties (F : SpecialFunction) :
  (F.f 0 = 2) ∧
  (∀ x, F.f x = F.f (-x)) ∧
  (∀ x, F.f (2 * x) = F.f x) := by
  sorry


end special_function_properties_l4037_403754


namespace shopping_solution_l4037_403722

/-- Represents the shopping problem with given prices, discounts, and taxes -/
def shopping_problem (initial_amount : ℝ) 
  (milk_price bread_price detergent_price banana_price_per_pound egg_price chicken_price apple_price : ℝ)
  (detergent_discount chicken_discount loyalty_discount milk_discount bread_discount : ℝ)
  (sales_tax : ℝ) : Prop :=
  let discounted_milk := milk_price * (1 - milk_discount)
  let discounted_bread := bread_price * (1 + 0.5) -- Buy one get one 50% off
  let discounted_detergent := detergent_price - detergent_discount
  let banana_total := banana_price_per_pound * 3
  let discounted_chicken := chicken_price * (1 - chicken_discount)
  let subtotal := discounted_milk + discounted_bread + discounted_detergent + banana_total + 
                  egg_price + discounted_chicken + apple_price
  let loyalty_discounted := subtotal * (1 - loyalty_discount)
  let total_with_tax := loyalty_discounted * (1 + sales_tax)
  initial_amount - total_with_tax = 38.25

/-- Theorem stating the solution to the shopping problem -/
theorem shopping_solution : 
  shopping_problem 75 3.80 4.25 11.50 0.95 2.80 8.45 6.30 2 0.20 0.10 0.15 0.50 0.08 := by
  sorry

end shopping_solution_l4037_403722


namespace smallest_multiple_year_l4037_403766

def joey_age : ℕ := 40
def chloe_age : ℕ := 38
def father_age : ℕ := 60

theorem smallest_multiple_year : 
  ∃ (n : ℕ), n > 0 ∧ 
  (joey_age + n) % father_age = 0 ∧ 
  (chloe_age + n) % father_age = 0 ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    (joey_age + m) % father_age ≠ 0 ∨ 
    (chloe_age + m) % father_age ≠ 0) ∧
  n = 180 :=
by sorry

end smallest_multiple_year_l4037_403766


namespace partial_fraction_decomposition_l4037_403743

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x^2 - 20 = A*(x+2)*(x-3) + B*(x-2)*(x-3) + C*(x-2)*(x+2)) →
  A * B * C = 2816 / 35 := by
sorry

end partial_fraction_decomposition_l4037_403743


namespace quadratic_equation_condition_l4037_403732

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, (a - 3) * x^2 - 4*x + 1 = 0 → (a - 3) ≠ 0) ↔ a ≠ 3 := by sorry

end quadratic_equation_condition_l4037_403732


namespace plane_Q_satisfies_conditions_l4037_403709

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + y - z - 6
def planeQ (x y z : ℝ) : ℝ := 19*x - 67*y + 109*z - 362

def point : ℝ × ℝ × ℝ := (2, 0, 3)

theorem plane_Q_satisfies_conditions :
  (∀ x y z : ℝ, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeQ x y z = 0) ∧ 
  (planeQ ≠ plane1 ∧ planeQ ≠ plane2) ∧
  (let (x₀, y₀, z₀) := point
   abs (19*x₀ - 67*y₀ + 109*z₀ - 362) / Real.sqrt (19^2 + (-67)^2 + 109^2) = 3 / Real.sqrt 2) :=
by sorry

end plane_Q_satisfies_conditions_l4037_403709


namespace geometric_sequence_sum_terms_l4037_403781

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The number of terms in the geometric sequence with first term 1/3 and common ratio 1/2, 
    whose sum equals 80/243, is 4 -/
theorem geometric_sequence_sum_terms : 
  ∃ (n : ℕ), n = 4 ∧ geometricSum (1/3) (1/2) n = 80/243 :=
sorry

end geometric_sequence_sum_terms_l4037_403781


namespace erased_number_proof_l4037_403724

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n ∧ x ≥ 1 →
  (n * (n + 1) / 2 - x) / (n - 1) = 614 / 17 →
  x = 7 := by
sorry

end erased_number_proof_l4037_403724


namespace triangle_equations_l4037_403763

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)  -- Side lengths
  (A B C : ℝ)  -- Angles in radians
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- Side lengths are positive
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)  -- Angles are positive
  (h7 : A + B + C = π)  -- Sum of angles is π

-- Define the theorem
theorem triangle_equations (t : Triangle) (h : t.A = π/3) :
  t.a * Real.sin t.C - Real.sqrt 3 * t.c * Real.cos t.A = 0 ∧
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) ∧
  Real.sqrt 3 * t.b * Real.sin t.A - t.a * Real.cos t.C = (t.c + t.b) * Real.cos t.A :=
by sorry

end triangle_equations_l4037_403763


namespace product_of_three_digit_numbers_l4037_403736

theorem product_of_three_digit_numbers : ∃ (I K S : Nat), 
  (I ≠ 0 ∧ K ≠ 0 ∧ S ≠ 0) ∧  -- non-zero digits
  (I ≠ K ∧ K ≠ S ∧ I ≠ S) ∧  -- distinct digits
  (I < 10 ∧ K < 10 ∧ S < 10) ∧  -- single digits
  ((100 * I + 10 * K + S) * (100 * K + 10 * S + I) = 100602) ∧  -- product
  (100602 % 10 = S) ∧  -- ends with S
  (100602 / 100 = I * 10 + K) ∧  -- after removing zeros, IKS remains
  (S = 2 ∧ K = 6 ∧ I = 1)  -- specific values that satisfy the conditions
:= by sorry

end product_of_three_digit_numbers_l4037_403736


namespace fixed_points_of_f_l4037_403720

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

theorem fixed_points_of_f (a b : ℝ) (ha : a ≠ 0) :
  -- Part 1
  (a = 1 ∧ b = 2 → ∃ x : ℝ, f 1 2 x = x ∧ x = -1) ∧
  -- Part 2
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = x₁ ∧ f a b x₂ = x₂) ↔ 0 < a ∧ a < 1) ∧
  -- Part 3
  (0 < a ∧ a < 1 →
    ∀ x₁ x₂ : ℝ, f a b x₁ = x₁ → f a b x₂ = x₂ →
      f a b x₁ + x₂ = -a / (2 * a^2 + 1) →
        0 < b ∧ b < 1/3) :=
by sorry

end fixed_points_of_f_l4037_403720


namespace equation_D_is_correct_l4037_403770

theorem equation_D_is_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 := by
  sorry

end equation_D_is_correct_l4037_403770


namespace amount_after_two_years_l4037_403700

theorem amount_after_two_years (initial_amount : ℝ) : 
  initial_amount = 6400 →
  (initial_amount * (81 / 64) : ℝ) = 8100 := by
sorry

end amount_after_two_years_l4037_403700


namespace distance_calculation_l4037_403773

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

/-- Distance from Xiaoqing's home to the park in meters -/
def total_distance : ℝ := 6000

/-- Distance Xiaoqing has already walked in meters -/
def walked_distance : ℝ := 1200

/-- Theorem stating the conversion of total distance to kilometers and the remaining distance to the park -/
theorem distance_calculation :
  (total_distance / meters_to_km = 6) ∧
  (total_distance - walked_distance = 4800) := by
  sorry

end distance_calculation_l4037_403773


namespace least_distinct_values_l4037_403761

theorem least_distinct_values (n : ℕ) (mode_freq : ℕ) (list_size : ℕ) 
  (h1 : n > 0)
  (h2 : mode_freq = 13)
  (h3 : list_size = 2023) :
  (∃ (list : List ℕ),
    list.length = list_size ∧
    (∃ (mode : ℕ), list.count mode = mode_freq ∧
      ∀ x : ℕ, x ≠ mode → list.count x < mode_freq) ∧
    (∀ m : ℕ, m < n → ¬∃ (list' : List ℕ),
      list'.length = list_size ∧
      (∃ (mode' : ℕ), list'.count mode' = mode_freq ∧
        ∀ x : ℕ, x ≠ mode' → list'.count x < mode_freq) ∧
      list'.toFinset.card = m)) →
  n = 169 := by
sorry

end least_distinct_values_l4037_403761


namespace area_of_removed_triangles_l4037_403740

theorem area_of_removed_triangles (side_length : ℝ) (hypotenuse : ℝ) : 
  side_length = 16 → hypotenuse = 8 → 
  4 * (1/2 * (hypotenuse^2 / 2)) = 64 := by
  sorry

end area_of_removed_triangles_l4037_403740


namespace ace_ten_jack_prob_is_16_33150_l4037_403782

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (aces : Nat)
  (tens : Nat)
  (jacks : Nat)

/-- Calculates the probability of drawing a specific card from the deck -/
def draw_probability (deck : Deck) (target_cards : Nat) : Rat :=
  target_cards / deck.total_cards

/-- Calculates the probability of drawing an Ace, then a 10, then a Jack -/
def ace_ten_jack_probability (deck : Deck) : Rat :=
  let p1 := draw_probability deck deck.aces
  let p2 := draw_probability { deck with total_cards := deck.total_cards - 1 } deck.tens
  let p3 := draw_probability { deck with total_cards := deck.total_cards - 2 } deck.jacks
  p1 * p2 * p3

/-- The main theorem to be proved -/
theorem ace_ten_jack_prob_is_16_33150 :
  let standard_deck : Deck := { total_cards := 52, aces := 4, tens := 4, jacks := 4 }
  ace_ten_jack_probability standard_deck = 16 / 33150 := by
  sorry

end ace_ten_jack_prob_is_16_33150_l4037_403782


namespace pair_probability_after_removal_l4037_403741

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_fives : ℕ)
  (num_threes : ℕ)

/-- Calculates the number of ways to choose 2 cards from a deck -/
def choose_two (d : Deck) : ℕ := Nat.choose d.size 2

/-- Calculates the number of ways to form pairs in a deck -/
def num_pairs (d : Deck) : ℕ := d.num_fives * Nat.choose 5 2 + d.num_threes * Nat.choose 3 2

/-- The probability of selecting a pair from the deck -/
def pair_probability (d : Deck) : ℚ := (num_pairs d : ℚ) / (choose_two d : ℚ)

theorem pair_probability_after_removal :
  let d : Deck := ⟨46, 4, 2⟩
  pair_probability d = 46 / 1035 :=
sorry

end pair_probability_after_removal_l4037_403741


namespace annual_income_difference_l4037_403778

/-- Given an 8% raise, if a person's raise is Rs. 800 and another person's raise is Rs. 840,
    then the difference between their new annual incomes is Rs. 540. -/
theorem annual_income_difference (D W : ℝ) : 
  0.08 * D = 800 → 0.08 * W = 840 → W + 840 - (D + 800) = 540 := by
  sorry

end annual_income_difference_l4037_403778


namespace dubblefud_game_l4037_403742

theorem dubblefud_game (red_value blue_value green_value : ℕ)
  (total_product : ℕ) (red blue green : ℕ) :
  red_value = 3 →
  blue_value = 7 →
  green_value = 11 →
  total_product = 5764801 →
  blue = green →
  (red_value ^ red) * (blue_value ^ blue) * (green_value ^ green) = total_product →
  red = 7 := by
  sorry

end dubblefud_game_l4037_403742


namespace complete_square_transformation_l4037_403795

theorem complete_square_transformation (x : ℝ) : 
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by
  sorry

end complete_square_transformation_l4037_403795


namespace intersection_M_N_l4037_403757

def M : Set ℝ := {y | ∃ x, y = x^2}

def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

theorem intersection_M_N :
  (M.prod Set.univ) ∩ N = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ Real.sqrt 2 ∧ p.2 = p.1^2} := by sorry

end intersection_M_N_l4037_403757


namespace intersection_of_A_and_B_l4037_403762

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-2) (-1) := by sorry

end intersection_of_A_and_B_l4037_403762


namespace right_triangle_identification_l4037_403733

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 5 12 13) ∧
  (¬ is_right_triangle 6 8 12) ∧
  (¬ is_right_triangle 6 12 15) :=
sorry

end right_triangle_identification_l4037_403733


namespace octagon_diagonals_l4037_403745

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l4037_403745


namespace rectangle_width_l4037_403734

/-- Given a rectangular area with a known area and length, prove that its width is 7 feet. -/
theorem rectangle_width (area : ℝ) (length : ℝ) (h1 : area = 35) (h2 : length = 5) :
  area / length = 7 := by
  sorry

end rectangle_width_l4037_403734


namespace fraction_value_theorem_l4037_403783

theorem fraction_value_theorem (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  (a - b) / (a + b) = -7 ∨ (a - b) / (a + b) = -1/7 := by
  sorry

end fraction_value_theorem_l4037_403783


namespace quadratic_roots_to_coefficients_l4037_403730

theorem quadratic_roots_to_coefficients (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -3) → 
  b = 1 ∧ c = -6 :=
by sorry

end quadratic_roots_to_coefficients_l4037_403730


namespace two_x_plus_three_equals_nine_l4037_403725

theorem two_x_plus_three_equals_nine (x : ℝ) (h : x = 3) : 2 * x + 3 = 9 := by
  sorry

end two_x_plus_three_equals_nine_l4037_403725


namespace nanometer_to_meter_one_nanometer_def_l4037_403706

/-- Proves that 28 nanometers is equal to 2.8 × 10^(-8) meters. -/
theorem nanometer_to_meter : 
  (28 : ℝ) * (1e-9 : ℝ) = (2.8 : ℝ) * (1e-8 : ℝ) := by
  sorry

/-- Defines the conversion factor from nanometers to meters. -/
def nanometer_to_meter_conversion : ℝ := 1e-9

/-- Proves that 1 nanometer is equal to 10^(-9) meters. -/
theorem one_nanometer_def : 
  (1 : ℝ) * nanometer_to_meter_conversion = (1e-9 : ℝ) := by
  sorry

end nanometer_to_meter_one_nanometer_def_l4037_403706


namespace circle_properties_l4037_403767

/-- A circle described by the equation x^2 + y^2 + 2ax - 2ay = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*a*p.1 - 2*a*p.2 = 0}

/-- The line x + y = 0 -/
def SymmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 0}

theorem circle_properties (a : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ Circle a ↔ (-y, -x) ∈ Circle a) ∧ 
  (0, 0) ∈ Circle a := by
  sorry

end circle_properties_l4037_403767


namespace fabric_cost_difference_l4037_403786

/-- The amount of fabric Kenneth bought in ounces -/
def kenneth_fabric : ℝ := 700

/-- The price per ounce of fabric in dollars -/
def price_per_oz : ℝ := 40

/-- The amount of fabric Nicholas bought in ounces -/
def nicholas_fabric : ℝ := 6 * kenneth_fabric

/-- The total cost of Kenneth's fabric in dollars -/
def kenneth_cost : ℝ := kenneth_fabric * price_per_oz

/-- The total cost of Nicholas's fabric in dollars -/
def nicholas_cost : ℝ := nicholas_fabric * price_per_oz

/-- The difference in cost between Nicholas's and Kenneth's fabric purchases -/
theorem fabric_cost_difference : nicholas_cost - kenneth_cost = 140000 := by
  sorry

end fabric_cost_difference_l4037_403786


namespace total_balls_in_bag_l4037_403701

/-- The number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ
  purple : ℕ

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple (bag : BagContents) : ℚ :=
  (bag.white + bag.green + bag.yellow : ℚ) / (bag.white + bag.green + bag.yellow + bag.red + bag.purple)

/-- The theorem stating the total number of balls in the bag -/
theorem total_balls_in_bag (bag : BagContents) 
  (h1 : bag.white = 10)
  (h2 : bag.green = 30)
  (h3 : bag.yellow = 10)
  (h4 : bag.red = 47)
  (h5 : bag.purple = 3)
  (h6 : prob_not_red_or_purple bag = 1/2) :
  bag.white + bag.green + bag.yellow + bag.red + bag.purple = 100 := by
  sorry

end total_balls_in_bag_l4037_403701


namespace lindas_furniture_spending_l4037_403772

theorem lindas_furniture_spending (savings : ℚ) (tv_cost : ℚ) (furniture_fraction : ℚ) :
  savings = 840 →
  tv_cost = 210 →
  furniture_fraction * savings + tv_cost = savings →
  furniture_fraction = 3/4 := by
sorry

end lindas_furniture_spending_l4037_403772


namespace average_expenditure_feb_to_july_l4037_403768

/-- Calculates the average expenditure for February to July given the conditions -/
theorem average_expenditure_feb_to_july 
  (avg_jan_to_june : ℝ) 
  (expenditure_jan : ℝ) 
  (expenditure_july : ℝ) 
  (h1 : avg_jan_to_june = 4200)
  (h2 : expenditure_jan = 1200)
  (h3 : expenditure_july = 1500) :
  (6 * avg_jan_to_june - expenditure_jan + expenditure_july) / 6 = 4250 := by
  sorry

#check average_expenditure_feb_to_july

end average_expenditure_feb_to_july_l4037_403768


namespace triangle_heights_theorem_l4037_403793

/-- A triangle with given heights -/
structure Triangle where
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Definition of an acute triangle based on heights -/
def is_acute (t : Triangle) : Prop :=
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧ t.ha ≠ t.hb ∧ t.hb ≠ t.hc ∧ t.ha ≠ t.hc

/-- Definition of triangle existence based on heights -/
def triangle_exists (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    t.ha = (2 * (a * b * c) / (a * b + b * c + c * a)) / a ∧
    t.hb = (2 * (a * b * c) / (a * b + b * c + c * a)) / b ∧
    t.hc = (2 * (a * b * c) / (a * b + b * c + c * a)) / c

theorem triangle_heights_theorem :
  (let t1 : Triangle := ⟨4, 5, 6⟩
   is_acute t1) ∧
  (let t2 : Triangle := ⟨2, 3, 6⟩
   ¬ triangle_exists t2) := by
  sorry

end triangle_heights_theorem_l4037_403793


namespace shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l4037_403703

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define median, angle bisector, and altitude
def median (t : Triangle) : ℝ := sorry
def angle_bisector (t : Triangle) : ℝ := sorry
def altitude (t : Triangle) : ℝ := sorry

-- Theorem 1: The shortest median is never longer than the longest angle bisector
theorem shortest_median_not_longer_than_longest_bisector (t : Triangle) :
  ∀ m b, median t ≤ m → angle_bisector t ≥ b → m ≤ b :=
sorry

-- Theorem 2: The shortest angle bisector is never longer than the longest altitude
theorem shortest_bisector_not_longer_than_longest_altitude (t : Triangle) :
  ∀ b h, angle_bisector t ≤ b → altitude t ≥ h → b ≤ h :=
sorry

end shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l4037_403703


namespace cake_distribution_l4037_403705

theorem cake_distribution (n : ℕ) (most least : ℚ) : 
  most = 1/11 → least = 1/14 → (∀ x, least ≤ x ∧ x ≤ most) → 
  (n : ℚ) * least ≤ 1 ∧ 1 ≤ (n : ℚ) * most → n = 12 ∨ n = 13 := by
  sorry

#check cake_distribution

end cake_distribution_l4037_403705


namespace possibly_six_l4037_403789

/-- Represents the possible outcomes of a dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- A fair six-sided dice -/
structure FairDice :=
  (outcomes : Finset DiceOutcome)
  (fair : outcomes.card = 6)
  (complete : ∀ o : DiceOutcome, o ∈ outcomes)

/-- The result of a single throw of a fair dice -/
def singleThrow (d : FairDice) : Set DiceOutcome :=
  d.outcomes

theorem possibly_six (d : FairDice) : 
  DiceOutcome.six ∈ singleThrow d :=
sorry

end possibly_six_l4037_403789


namespace percentage_value_in_quarters_l4037_403790

/-- Represents the number of nickels --/
def num_nickels : ℕ := 80

/-- Represents the number of quarters --/
def num_quarters : ℕ := 40

/-- Represents the value of a nickel in cents --/
def nickel_value : ℕ := 5

/-- Represents the value of a quarter in cents --/
def quarter_value : ℕ := 25

/-- Theorem stating that the percentage of total value in quarters is 5/7 --/
theorem percentage_value_in_quarters :
  (num_quarters * quarter_value : ℚ) / (num_nickels * nickel_value + num_quarters * quarter_value) = 5 / 7 := by
  sorry

end percentage_value_in_quarters_l4037_403790


namespace existence_of_multiple_factorizations_l4037_403735

/-- The set V_n of integers of the form 1 + kn where k ≥ 1 -/
def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

/-- A number is indecomposable in V_n if it can't be expressed as a product of two numbers from V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → m ≠ p * q

/-- Two lists of natural numbers are considered different if they are not permutations of each other -/
def DifferentFactorizations (l1 l2 : List ℕ) : Prop :=
  ¬(l1.Perm l2)

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ l1 l2 : List ℕ,
      (∀ x ∈ l1, Indecomposable n x) ∧
      (∀ x ∈ l2, Indecomposable n x) ∧
      (r = l1.prod) ∧
      (r = l2.prod) ∧
      DifferentFactorizations l1 l2 :=
sorry


end existence_of_multiple_factorizations_l4037_403735


namespace alcohol_concentration_reduction_specific_alcohol_reduction_l4037_403753

/-- Calculates the percentage reduction in alcohol concentration when water is added to an alcohol solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific case of adding 26 liters of water to 14 liters of 20% alcohol solution results in a 65% reduction in concentration. -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 14 0.20 26 = 65 := by
  -- Proof goes here
  sorry

end alcohol_concentration_reduction_specific_alcohol_reduction_l4037_403753


namespace k_value_l4037_403721

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem k_value : ∀ k : ℕ, A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end k_value_l4037_403721


namespace interest_rate_calculation_l4037_403794

/-- Represents the interest rate calculation problem --/
theorem interest_rate_calculation 
  (principal : ℝ) 
  (amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 896) 
  (h2 : amount = 1120) 
  (h3 : time = 5) :
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end interest_rate_calculation_l4037_403794


namespace quadratic_root_triple_relation_l4037_403737

theorem quadratic_root_triple_relation (a b c : ℝ) :
  (∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ 
              a * y^2 + b * y + c = 0 ∧ 
              y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end quadratic_root_triple_relation_l4037_403737


namespace log_sum_property_l4037_403765

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end log_sum_property_l4037_403765


namespace prob_white_glow_pop_is_12_21_l4037_403791

/-- Represents the color of a kernel -/
inductive KernelColor
| White
| Yellow

/-- Represents the properties of kernels in the bag -/
structure KernelProperties where
  totalWhite : Rat
  totalYellow : Rat
  whiteGlow : Rat
  yellowGlow : Rat
  whiteGlowPop : Rat
  yellowGlowPop : Rat

/-- The given properties of the kernels in the bag -/
def bagProperties : KernelProperties :=
  { totalWhite := 3/4
  , totalYellow := 1/4
  , whiteGlow := 1/2
  , yellowGlow := 3/4
  , whiteGlowPop := 1/2
  , yellowGlowPop := 3/4
  }

/-- The probability that a randomly selected kernel that glows and pops is white -/
def probWhiteGlowPop (props : KernelProperties) : Rat :=
  let whiteGlowPop := props.totalWhite * props.whiteGlow * props.whiteGlowPop
  let yellowGlowPop := props.totalYellow * props.yellowGlow * props.yellowGlowPop
  whiteGlowPop / (whiteGlowPop + yellowGlowPop)

/-- Theorem stating that the probability of selecting a white kernel that glows and pops is 12/21 -/
theorem prob_white_glow_pop_is_12_21 :
  probWhiteGlowPop bagProperties = 12/21 := by
  sorry

end prob_white_glow_pop_is_12_21_l4037_403791


namespace square_pyramid_components_l4037_403707

/-- The number of rows in the square pyramid -/
def num_rows : ℕ := 10

/-- The number of unit rods in the first row -/
def first_row_rods : ℕ := 4

/-- The number of additional rods in each subsequent row -/
def additional_rods_per_row : ℕ := 4

/-- Calculate the total number of unit rods in the pyramid -/
def total_rods (n : ℕ) : ℕ :=
  first_row_rods * n * (n + 1) / 2

/-- Calculate the number of internal connectors -/
def internal_connectors (n : ℕ) : ℕ :=
  4 * (n * (n - 1) / 2)

/-- Calculate the number of vertical connectors -/
def vertical_connectors (n : ℕ) : ℕ :=
  4 * (n - 1)

/-- The total number of connectors -/
def total_connectors (n : ℕ) : ℕ :=
  internal_connectors n + vertical_connectors n

/-- The main theorem: proving the total number of unit rods and connectors -/
theorem square_pyramid_components :
  total_rods num_rows + total_connectors num_rows = 436 := by
  sorry

end square_pyramid_components_l4037_403707


namespace cone_sphere_ratio_l4037_403726

/-- Theorem: For a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the altitude of the cone to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * π * r^3) = (1 / 3) * π * r^2 * h →
  h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l4037_403726


namespace ellipse_equation_from_conditions_l4037_403785

/-- Represents an ellipse with axes of symmetry on the coordinate axes -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions :
  ∀ e : Ellipse,
    e.a + e.b = 9 →  -- Sum of semi-axes is 9 (half of 18)
    e.a^2 - e.b^2 = 9 →  -- Focal distance squared is 9 (6^2 / 4)
    (∀ x y : ℝ, ellipse_equation e x y ↔ 
      (x^2 / 25 + y^2 / 16 = 1 ∨ x^2 / 16 + y^2 / 25 = 1)) :=
by
  sorry

end ellipse_equation_from_conditions_l4037_403785


namespace three_sevenths_decomposition_l4037_403784

theorem three_sevenths_decomposition :
  3 / 7 = 1 / 8 + 1 / 56 + 1 / 9 + 1 / 72 := by
  sorry

#check three_sevenths_decomposition

end three_sevenths_decomposition_l4037_403784


namespace min_k_good_is_two_l4037_403779

/-- A function f: ℕ+ → ℕ+ is k-good if for all m ≠ n in ℕ+, (f(m)+n, f(n)+m) ≤ k -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- The minimum k for which a k-good function exists is 2 -/
theorem min_k_good_is_two :
  (∃ k : ℕ, k > 0 ∧ ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (∀ k : ℕ, k > 0 → (∃ f : ℕ+ → ℕ+, IsKGood k f) → k ≥ 2) :=
by sorry

#check min_k_good_is_two

end min_k_good_is_two_l4037_403779


namespace vector_relations_l4037_403788

def a : Fin 3 → ℝ
| 0 => 2
| 1 => -1
| 2 => 3
| _ => 0

def b (x : ℝ) : Fin 3 → ℝ
| 0 => -4
| 1 => 2
| 2 => x
| _ => 0

theorem vector_relations (x : ℝ) :
  (∀ i : Fin 3, (a i) * (b x i) = 0 → x = 10/3) ∧
  (∃ k : ℝ, ∀ i : Fin 3, (a i) = k * (b x i) → x = -6) := by
  sorry

end vector_relations_l4037_403788


namespace ages_sum_l4037_403780

theorem ages_sum (j l : ℝ) : 
  j = l + 8 ∧ 
  j + 5 = 3 * (l - 6) → 
  j + l = 39 := by
  sorry

end ages_sum_l4037_403780


namespace parabola_hyperbola_focus_coincide_l4037_403731

/-- The value of p for which the focus of the parabola x^2 = 2py (p > 0) 
    coincides with the focus of the hyperbola y^2/3 - x^2 = 1 -/
theorem parabola_hyperbola_focus_coincide : 
  ∃ p : ℝ, p > 0 ∧ 
  (∀ x y : ℝ, x^2 = 2*p*y ↔ (x, y) ∈ {(x, y) | x^2 = 2*p*y}) ∧
  (∀ x y : ℝ, y^2/3 - x^2 = 1 ↔ (x, y) ∈ {(x, y) | y^2/3 - x^2 = 1}) ∧
  (0, p/2) = (0, 2) ∧
  p = 4 :=
sorry

end parabola_hyperbola_focus_coincide_l4037_403731


namespace complex_moduli_sum_l4037_403760

theorem complex_moduli_sum : 
  let z1 : ℂ := 3 - 5*I
  let z2 : ℂ := 3 + 5*I
  Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := by
sorry

end complex_moduli_sum_l4037_403760


namespace supplement_of_supplement_35_l4037_403748

/-- The supplement of an angle is the angle that, when added to the original angle, forms a straight angle (180 degrees). -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- Theorem: The supplement of the supplement of a 35-degree angle is 35 degrees. -/
theorem supplement_of_supplement_35 :
  supplement (supplement 35) = 35 := by
  sorry

end supplement_of_supplement_35_l4037_403748


namespace passing_marks_calculation_l4037_403774

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.35 * T = P - 40) → 
  (0.60 * T = P + 25) → 
  P = 131 := by
  sorry

end passing_marks_calculation_l4037_403774


namespace mn_value_l4037_403739

theorem mn_value (M N : ℝ) 
  (h1 : (Real.log N) / (2 * Real.log M) = 2 * Real.log M / Real.log N)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = Real.sqrt N :=
by sorry

end mn_value_l4037_403739


namespace fibonacci_fifth_divisible_by_five_l4037_403711

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fifth_divisible_by_five (k : ℕ) :
  5 ∣ fibonacci (5 * k) := by
  sorry

end fibonacci_fifth_divisible_by_five_l4037_403711


namespace xy_divides_x_squared_plus_2y_minus_1_l4037_403796

theorem xy_divides_x_squared_plus_2y_minus_1 (x y : ℕ+) :
  (x * y) ∣ (x^2 + 2*y - 1) ↔ 
  ((x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8) ∨ 
   (x = 1) ∨ 
   (∃ n : ℕ+, x = 2*n - 1 ∧ y = n)) := by
sorry

end xy_divides_x_squared_plus_2y_minus_1_l4037_403796


namespace blue_marble_probability_l4037_403750

theorem blue_marble_probability
  (total_marbles : ℕ)
  (red_prob : ℚ)
  (h1 : total_marbles = 30)
  (h2 : red_prob = 32/75) :
  ∃ (x y : ℕ) (r1 r2 : ℕ),
    x + y = total_marbles ∧
    (r1 : ℚ) * r2 / (x * y) = red_prob ∧
    ((x - r1) : ℚ) * (y - r2) / (x * y) = 3/25 := by
  sorry

#eval 3 + 25  -- Expected output: 28

end blue_marble_probability_l4037_403750


namespace toms_seashells_l4037_403749

/-- Calculates the number of unbroken seashells Tom had left after three days of collecting and giving some away. -/
theorem toms_seashells (day1_total day1_broken day2_total day2_broken day3_total day3_broken given_away : ℕ) 
  (h1 : day1_total = 7)
  (h2 : day1_broken = 4)
  (h3 : day2_total = 12)
  (h4 : day2_broken = 5)
  (h5 : day3_total = 15)
  (h6 : day3_broken = 8)
  (h7 : given_away = 3) :
  day1_total - day1_broken + day2_total - day2_broken + day3_total - day3_broken - given_away = 14 := by
  sorry


end toms_seashells_l4037_403749


namespace game_winning_strategy_l4037_403728

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the result of the game -/
inductive GameResult
| AWins
| BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  k : ℕ
  grid : Fin n → Fin n → Bool
  currentPlayer : Player

/-- Defines the winning strategy for the game -/
def winningStrategy (n k : ℕ) : GameResult :=
  if n ≤ 2 * k - 1 then
    GameResult.AWins
  else if n % 2 = 1 then
    GameResult.AWins
  else
    GameResult.BWins

/-- The main theorem stating the winning strategy for the game -/
theorem game_winning_strategy (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  (winningStrategy n k = GameResult.AWins ∧ 
   (n ≤ 2 * k - 1 ∨ (n ≥ 2 * k ∧ n % 2 = 1))) ∨
  (winningStrategy n k = GameResult.BWins ∧ 
   n ≥ 2 * k ∧ n % 2 = 0) :=
by sorry


end game_winning_strategy_l4037_403728


namespace ceiling_floor_difference_l4037_403744

theorem ceiling_floor_difference : 
  ⌈(10 : ℝ) / 4 * (-17 : ℝ) / 2⌉ - ⌊(10 : ℝ) / 4 * ⌊(-17 : ℝ) / 2⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l4037_403744


namespace count_values_for_sum_20_main_theorem_l4037_403764

def count_integer_values (n : ℕ) : ℕ :=
  (Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))).card

theorem count_values_for_sum_20 :
  count_integer_values 20 = 6 :=
sorry

theorem main_theorem :
  ∃ (S : Finset ℤ),
    S.card = 6 ∧
    ∀ (a b c : ℕ),
      a > 0 → b > 0 → c > 0 →
      a + b + c = 20 →
      (a + b : ℤ) / (c : ℤ) ∈ S :=
sorry

end count_values_for_sum_20_main_theorem_l4037_403764


namespace three_lines_two_intersections_l4037_403751

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define what it means for two points to be distinct
def distinct (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

-- Define what it means for a point to be on a line
def on_line1 (p : ℝ × ℝ) : Prop := line1 p.1 p.2
def on_line2 (p : ℝ × ℝ) : Prop := line2 p.1 p.2
def on_line3 (a : ℝ) (p : ℝ × ℝ) : Prop := line3 a p.1 p.2

-- Theorem statement
theorem three_lines_two_intersections (a : ℝ) :
  (∃ p1 p2 : ℝ × ℝ, distinct p1 p2 ∧ 
    on_line1 p1 ∧ on_line1 p2 ∧ 
    on_line2 p1 ∧ on_line2 p2 ∧ 
    on_line3 a p1 ∧ on_line3 a p2 ∧
    (∀ p3 : ℝ × ℝ, on_line1 p3 ∧ on_line2 p3 ∧ on_line3 a p3 → p3 = p1 ∨ p3 = p2)) →
  a = 3 ∨ a = -6 :=
sorry

end three_lines_two_intersections_l4037_403751


namespace quadrilateral_equal_area_implies_midpoint_l4037_403769

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (E : Point) (quad : Quadrilateral) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

theorem quadrilateral_equal_area_implies_midpoint 
  (quad : Quadrilateral) (E : Point) :
  isInside E quad →
  (triangleArea E quad.A quad.B = triangleArea E quad.B quad.C) ∧
  (triangleArea E quad.B quad.C = triangleArea E quad.C quad.D) ∧
  (triangleArea E quad.C quad.D = triangleArea E quad.D quad.A) →
  (isMidpoint E quad.A quad.C) ∨ (isMidpoint E quad.B quad.D) := by
  sorry

end quadrilateral_equal_area_implies_midpoint_l4037_403769


namespace equation_solution_l4037_403746

theorem equation_solution : ∃! x : ℚ, (3 / 20 + 3 / x = 8 / x + 1 / 15) ∧ x = 60 := by sorry

end equation_solution_l4037_403746


namespace line_through_points_l4037_403713

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a given line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The theorem states that the line x - 2y + 1 = 0 passes through points A(-1, 0) and B(3, 2) -/
theorem line_through_points :
  let A : Point2D := ⟨-1, 0⟩
  let B : Point2D := ⟨3, 2⟩
  let line : Line2D := ⟨1, -2, 1⟩
  point_on_line A line ∧ point_on_line B line :=
by sorry

end line_through_points_l4037_403713
