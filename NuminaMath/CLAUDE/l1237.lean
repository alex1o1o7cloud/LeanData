import Mathlib

namespace like_terms_exponent_sum_l1237_123759

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, a x y ≠ 0 ∧ b x y ≠ 0 → (a x y = b x y)

/-- The first monomial 5x^4y -/
def mono1 (x y : ℕ) : ℚ := 5 * x^4 * y

/-- The second monomial 5x^ny^m -/
def mono2 (n m x y : ℕ) : ℚ := 5 * x^n * y^m

theorem like_terms_exponent_sum :
  are_like_terms mono1 (mono2 n m) → n + m = 5 := by
  sorry

end like_terms_exponent_sum_l1237_123759


namespace triangle_properties_l1237_123750

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) :
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = π →
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c →
    t.B = π / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.A = 5 * π / 12 ∧ t.b = 2 →
    t.c = 2 * Real.sqrt 6 / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.a + t.c = 4 →
    ∀ x : ℝ, x > 0 → t.b ≤ x → 2 ≤ x) :=
by sorry

end triangle_properties_l1237_123750


namespace class_size_l1237_123773

theorem class_size : ∃ n : ℕ, 
  (20 < n ∧ n < 30) ∧ 
  (∃ x : ℕ, n = 3 * x) ∧ 
  (∃ y : ℕ, n = 4 * y - 1) ∧ 
  n = 27 := by
  sorry

end class_size_l1237_123773


namespace females_attending_correct_l1237_123716

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The total population of Nantucket -/
def total_population : ℕ := 300

/-- The number of people attending the meeting -/
def meeting_attendance : ℕ := total_population / 2

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

theorem females_attending_correct :
  females_attending = 50 ∧
  meeting_attendance = total_population / 2 ∧
  total_population = 300 ∧
  males_attending = 2 * females_attending ∧
  meeting_attendance = females_attending + males_attending :=
by sorry

end females_attending_correct_l1237_123716


namespace box_negative_two_zero_negative_one_l1237_123732

-- Define the box operation
def box (a b c : ℤ) : ℚ :=
  (a ^ b : ℚ) - if b = 0 ∧ c < 0 then 0 else (b ^ c : ℚ) + (c ^ a : ℚ)

-- State the theorem
theorem box_negative_two_zero_negative_one :
  box (-2) 0 (-1) = 2 := by sorry

end box_negative_two_zero_negative_one_l1237_123732


namespace berry_theorem_l1237_123701

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

theorem berry_theorem : berry_problem 26 10 9 = 7 := by
  sorry

end berry_theorem_l1237_123701


namespace sum_of_fractions_l1237_123784

theorem sum_of_fractions : 
  7/8 + 11/12 = 43/24 ∧ 
  (∀ n d : ℤ, (n ≠ 0 ∨ d ≠ 1) → 43 * d ≠ 24 * n) := by
  sorry

end sum_of_fractions_l1237_123784


namespace investment_ratio_l1237_123702

/-- Given two partners p and q, their profit ratio, and investment times, 
    prove the ratio of their investments. -/
theorem investment_ratio 
  (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (profit_ratio_constraint : profit_ratio_p / profit_ratio_q = 7 / 10)
  (time_constraint_p : investment_time_p = 8)
  (time_constraint_q : investment_time_q = 16) :
  ∃ (investment_p investment_q : ℚ),
    investment_p / investment_q = 7 / 5 ∧
    profit_ratio_p / profit_ratio_q = 
      (investment_p * investment_time_p) / (investment_q * investment_time_q) :=
by sorry

end investment_ratio_l1237_123702


namespace bisection_method_sign_l1237_123761

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval (a, b)
variable (a b : ℝ)

-- Define the sequence of intervals
variable (seq : ℕ → ℝ × ℝ)

-- State the theorem
theorem bisection_method_sign (hcont : Continuous f) 
  (hunique : ∃! x, x ∈ Ioo a b ∧ f x = 0)
  (hseq : ∀ k, Ioo (seq k).1 (seq k).2 ⊆ Ioo (seq (k+1)).1 (seq (k+1)).2)
  (hzero : ∀ k, ∃ x, x ∈ Ioo (seq k).1 (seq k).2 ∧ f x = 0)
  (hinit : seq 0 = (a, b))
  (hsign : f a < 0 ∧ f b > 0) :
  ∀ k, f (seq k).1 < 0 :=
sorry

end bisection_method_sign_l1237_123761


namespace group_collection_l1237_123719

/-- Calculates the total collection in rupees for a group contribution -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Theorem stating that for a group of 93 members, the total collection is 86.49 rupees -/
theorem group_collection :
  total_collection 93 = 86.49 := by
  sorry

end group_collection_l1237_123719


namespace fraction_equality_l1237_123735

theorem fraction_equality (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : (2/3) * x = k * (1/x)) : k = 2/27 := by
  sorry

end fraction_equality_l1237_123735


namespace parabola_square_min_area_l1237_123713

/-- A square in a Cartesian plane with vertices on two parabolas -/
structure ParabolaSquare where
  /-- x-coordinate of a vertex on y = x^2 -/
  a : ℝ
  /-- The square's side length -/
  s : ℝ
  /-- Two opposite vertices lie on y = x^2 -/
  h1 : (a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  h2 : (-a, a^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  /-- The other two opposite vertices lie on y = -x^2 + 4 -/
  h3 : (a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  h4 : (-a, -a^2 + 4) ∈ {p : ℝ × ℝ | p.2 = -p.1^2 + 4}
  /-- The side length is the distance between vertices -/
  h5 : s^2 = (2*a)^2 + (2*a^2 - 4)^2

/-- The smallest possible area of the ParabolaSquare is 4 -/
theorem parabola_square_min_area :
  ∀ (ps : ParabolaSquare), ∃ (min_ps : ParabolaSquare), min_ps.s^2 = 4 ∧ ∀ (ps' : ParabolaSquare), ps'.s^2 ≥ 4 :=
sorry

end parabola_square_min_area_l1237_123713


namespace hyperbola_equation_l1237_123741

/-- A hyperbola and a parabola sharing a common focus -/
structure HyperbolaParabola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  F : ℝ × ℝ
  P : ℝ × ℝ
  h_parabola : (P.2)^2 = 8 * P.1
  h_hyperbola : (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1
  h_common_focus : F = (2, 0)
  h_distance : Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5

/-- The equation of the hyperbola is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation (hp : HyperbolaParabola) : 
  hp.a = 1 ∧ hp.b = Real.sqrt 3 := by
  sorry

end hyperbola_equation_l1237_123741


namespace sin_alpha_plus_pi_12_l1237_123717

theorem sin_alpha_plus_pi_12 (α : ℝ) 
  (h1 : α ∈ Set.Ioo (-π/3) 0)
  (h2 : Real.cos (α + π/6) - Real.sin α = 4*Real.sqrt 3/5) :
  Real.sin (α + π/12) = -Real.sqrt 2/10 := by sorry

end sin_alpha_plus_pi_12_l1237_123717


namespace f_difference_at_five_l1237_123720

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

-- Theorem statement
theorem f_difference_at_five : f 5 - f (-5) = 800 := by
  sorry

end f_difference_at_five_l1237_123720


namespace max_ab_value_l1237_123766

theorem max_ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1 / 2) * Real.exp 3 := by
  sorry

end max_ab_value_l1237_123766


namespace sqrt_log_sum_equality_l1237_123797

theorem sqrt_log_sum_equality : 
  Real.sqrt (Real.log 6 / Real.log 2 + Real.log 6 / Real.log 3) = 
    Real.sqrt (Real.log 3 / Real.log 2) + Real.sqrt (Real.log 2 / Real.log 3) := by
  sorry

end sqrt_log_sum_equality_l1237_123797


namespace triangle_inequality_ratio_three_fourths_is_optimal_l1237_123730

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

-- Theorem statement
theorem triangle_inequality_ratio (t : Triangle) :
  (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 ≥ (3/4 : ℝ) :=
sorry

-- Theorem for the optimality of the bound
theorem three_fourths_is_optimal :
  ∀ ε > 0, ∃ t : Triangle, (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 < 3/4 + ε :=
sorry

end triangle_inequality_ratio_three_fourths_is_optimal_l1237_123730


namespace expression_equals_negative_one_l1237_123770

theorem expression_equals_negative_one (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x ∧ z ≠ -x) :
  (x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1 :=
sorry

end expression_equals_negative_one_l1237_123770


namespace polynomial_factorization_l1237_123731

theorem polynomial_factorization (x : ℝ) :
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 1)*(x^4 - 2*x^2 + 2) := by
  sorry

end polynomial_factorization_l1237_123731


namespace problem_solution_l1237_123746

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2*m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem problem_solution (m : ℝ) :
  (∀ x, x ∈ A ∩ B → x ∈ C m) →
  (B ∪ C m = Set.univ ∧ C m ⊆ D) →
  m ≥ 5/2 ∧ 7/2 ≤ m ∧ m < 4 := by
  sorry

end problem_solution_l1237_123746


namespace max_leftover_candies_l1237_123772

theorem max_leftover_candies (n : ℕ) : ∃ (k : ℕ), n = 8 * k + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end max_leftover_candies_l1237_123772


namespace diophantine_equation_solutions_l1237_123786

theorem diophantine_equation_solutions :
  let S : Set (ℕ × ℕ × ℕ × ℕ) := {(x, y, z, w) | 2^x * 3^y - 5^z * 7^w = 1}
  S = {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} := by
  sorry

end diophantine_equation_solutions_l1237_123786


namespace sum_of_three_numbers_l1237_123795

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 40)
  (sum3 : c + a = 45) :
  a + b + c = 60 := by
  sorry

end sum_of_three_numbers_l1237_123795


namespace normal_distribution_symmetry_l1237_123751

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value of 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

/-- Theorem stating that if P(X < a) = 0.2, then P(X < 4-a) = 0.2 -/
theorem normal_distribution_symmetry :
  (cdf_X a = 0.2) → (cdf_X (4 - a) = 0.2) := by
  sorry

end normal_distribution_symmetry_l1237_123751


namespace angle_in_second_quadrant_l1237_123794

theorem angle_in_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  |Real.cos (α / 3)| = -Real.cos (α / 3) →  -- |cos(α/3)| = -cos(α/3)
  (π / 2 < α / 3) ∧ (α / 3 < π)  -- α/3 is in the second quadrant
:= by sorry

end angle_in_second_quadrant_l1237_123794


namespace smaller_solution_quadratic_l1237_123745

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 7*x - 18 = 0) → (x = -2 ∨ x = 9) → -2 ≤ x :=
by sorry

end smaller_solution_quadratic_l1237_123745


namespace parabola_translation_l1237_123785

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-4) 0 0
  let translated := translate original 2 3
  y = -4 * x^2 → y = translated.a * (x + 2)^2 + translated.b * (x + 2) + translated.c :=
by sorry

end parabola_translation_l1237_123785


namespace swimmers_return_simultaneously_l1237_123762

/-- Represents a swimmer in the river scenario -/
structure Swimmer where
  speed : ℝ  -- Speed relative to water
  direction : Int  -- 1 for downstream, -1 for upstream

/-- Represents the river scenario -/
structure RiverScenario where
  current_speed : ℝ
  swim_time : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer

/-- Calculates the time taken for a swimmer to return to the raft -/
def return_time (scenario : RiverScenario) (swimmer : Swimmer) : ℝ :=
  2 * scenario.swim_time

theorem swimmers_return_simultaneously (scenario : RiverScenario) :
  return_time scenario scenario.swimmer1 = return_time scenario scenario.swimmer2 :=
sorry

end swimmers_return_simultaneously_l1237_123762


namespace unique_root_quadratic_l1237_123727

theorem unique_root_quadratic (c : ℝ) : 
  (∃ b : ℝ, b = c^2 + 1 ∧ 
   (∃! x : ℝ, x^2 + b*x + c = 0)) → 
  c = 1 := by
sorry

end unique_root_quadratic_l1237_123727


namespace min_product_under_constraints_l1237_123787

theorem min_product_under_constraints (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 2 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 32/81 :=
by sorry

end min_product_under_constraints_l1237_123787


namespace expression_bounds_l1237_123767

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 := by
  sorry

end expression_bounds_l1237_123767


namespace park_area_l1237_123729

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure Park where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park -/
def Park.perimeter (p : Park) : ℝ := 2 * (p.length + p.width)

/-- The area of the park -/
def Park.area (p : Park) : ℝ := p.length * p.width

/-- The cost of fencing per meter in rupees -/
def fencing_cost_per_meter : ℝ := 0.50

/-- The total cost of fencing the park in rupees -/
def total_fencing_cost : ℝ := 175

theorem park_area (p : Park) : 
  p.perimeter * fencing_cost_per_meter = total_fencing_cost → 
  p.area = 7350 := by
  sorry

#check park_area

end park_area_l1237_123729


namespace expression_simplification_l1237_123733

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1237_123733


namespace class_strength_solution_l1237_123754

/-- Represents the problem of finding the original class strength --/
def find_original_class_strength (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : Prop :=
  ∃ (original_strength : ℕ),
    (original_strength : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((original_strength : ℝ) + new_students) * (original_avg - avg_decrease)

/-- The theorem stating the solution to the class strength problem --/
theorem class_strength_solution : 
  find_original_class_strength 40 18 32 4 → 
  ∃ (original_strength : ℕ), original_strength = 18 :=
by
  sorry

#check class_strength_solution

end class_strength_solution_l1237_123754


namespace a_greater_than_b_l1237_123791

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end a_greater_than_b_l1237_123791


namespace regular_polygon_sides_l1237_123776

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides. -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) : 
  P = 180 → s = 15 → P = n * s → n = 12 := by sorry

end regular_polygon_sides_l1237_123776


namespace min_value_of_f_l1237_123704

/-- The quadratic function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = -44) := by
  sorry

end min_value_of_f_l1237_123704


namespace infinite_solutions_condition_l1237_123774

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ y : ℝ, y ≠ 0 → 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 2.5 := by
  sorry

end infinite_solutions_condition_l1237_123774


namespace infinite_primes_dividing_sequence_l1237_123744

theorem infinite_primes_dividing_sequence (a b c : ℕ) (ha : a ≠ c) (hb : b ≠ c) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ a^n + b^n - c^n} := by
  sorry

end infinite_primes_dividing_sequence_l1237_123744


namespace token_game_result_l1237_123710

def iterate_operation (n : ℕ) (f : ℤ → ℤ) (initial : ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_operation m f initial)

theorem token_game_result :
  let square (x : ℤ) := x * x
  let cube (x : ℤ) := x * x * x
  let iterations := 50
  let token1 := iterate_operation iterations square 2
  let token2 := iterate_operation iterations cube (-2)
  let token3 := iterate_operation iterations square 0
  token1 + token2 + token3 = -496 := by
  sorry

end token_game_result_l1237_123710


namespace snack_cost_per_person_l1237_123753

/-- Calculates the cost per person for a group of friends buying snacks -/
theorem snack_cost_per_person 
  (num_friends : ℕ) 
  (num_fish_cakes : ℕ) 
  (fish_cake_price : ℕ) 
  (num_tteokbokki : ℕ) 
  (tteokbokki_price : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : num_fish_cakes = 5) 
  (h3 : fish_cake_price = 200) 
  (h4 : num_tteokbokki = 7) 
  (h5 : tteokbokki_price = 800) : 
  (num_fish_cakes * fish_cake_price + num_tteokbokki * tteokbokki_price) / num_friends = 1650 := by
  sorry

end snack_cost_per_person_l1237_123753


namespace largest_two_digit_prime_factor_l1237_123778

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ 
  p ∣ binomial_coefficient 210 105 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 210 105 → q ≤ p ∧
  p = 67 :=
by sorry

end largest_two_digit_prime_factor_l1237_123778


namespace order_of_a_l1237_123788

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end order_of_a_l1237_123788


namespace nuts_to_raisins_cost_ratio_l1237_123771

/-- The ratio of the cost of nuts to raisins given the mixture proportions and cost ratio -/
theorem nuts_to_raisins_cost_ratio 
  (raisin_pounds : ℝ) 
  (nuts_pounds : ℝ)
  (raisin_cost : ℝ)
  (nuts_cost : ℝ)
  (h1 : raisin_pounds = 3)
  (h2 : nuts_pounds = 4)
  (h3 : raisin_cost > 0)
  (h4 : nuts_cost > 0)
  (h5 : raisin_pounds * raisin_cost = 0.15789473684210525 * (raisin_pounds * raisin_cost + nuts_pounds * nuts_cost)) :
  nuts_cost / raisin_cost = 4 := by
sorry

end nuts_to_raisins_cost_ratio_l1237_123771


namespace double_inequality_solution_l1237_123763

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 20*x + 21) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 20*x + 21) / (x^2 - 4*x + 5) < 1 ↔ 
  (2 < x ∧ x < 1) ∨ (26 < x) :=
sorry

end double_inequality_solution_l1237_123763


namespace school_c_variance_l1237_123711

/-- Represents the data for a school's strong math foundation group -/
structure SchoolData where
  students : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the overall data for all schools -/
structure OverallData where
  total_students : ℕ
  average : ℝ
  variance : ℝ

/-- Theorem stating that given the conditions, the variance of school C is 12 -/
theorem school_c_variance
  (ratio : Fin 3 → ℕ)
  (h_ratio : ratio = ![3, 2, 1])
  (overall : OverallData)
  (h_overall : overall = { total_students := 48, average := 117, variance := 21.5 })
  (school_a : SchoolData)
  (h_school_a : school_a = { students := 24, average := 118, variance := 15 })
  (school_b : SchoolData)
  (h_school_b : school_b = { students := 16, average := 114, variance := 21 })
  (school_c : SchoolData)
  (h_school_c_students : school_c.students = 8) :
  school_c.variance = 12 := by
  sorry

end school_c_variance_l1237_123711


namespace square_root_subtraction_l1237_123752

theorem square_root_subtraction : Real.sqrt 81 - Real.sqrt 144 * 3 = -27 := by
  sorry

end square_root_subtraction_l1237_123752


namespace floor_sqrt_120_l1237_123715

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l1237_123715


namespace complement_of_union_A_B_l1237_123792

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {0, 4, 5} := by sorry

end complement_of_union_A_B_l1237_123792


namespace green_tea_cost_july_l1237_123798

/-- Proves that the cost of green tea per pound in July is $0.30 --/
theorem green_tea_cost_july (june_cost : ℝ) : 
  (june_cost > 0) →  -- Assuming positive cost in June
  (june_cost + june_cost = 3.45 / 1.5) →  -- July mixture cost equation
  (0.3 * june_cost = 0.30) :=  -- July green tea cost
by
  sorry

#check green_tea_cost_july

end green_tea_cost_july_l1237_123798


namespace workshop_workers_l1237_123793

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 22

/-- The number of technicians in the workshop -/
def technicians : ℕ := 7

/-- The average salary of all workers -/
def avg_salary_all : ℚ := 850

/-- The average salary of technicians -/
def avg_salary_tech : ℚ := 1000

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 780

/-- Theorem stating that given the conditions, the total number of workers is 22 -/
theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) =
  (avg_salary_tech * technicians : ℚ) +
  (avg_salary_rest * (total_workers - technicians) : ℚ) :=
sorry

end workshop_workers_l1237_123793


namespace f_8_5_equals_1_5_l1237_123765

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_8_5_equals_1_5 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 3)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 3 * x) :
  f 8.5 = 1.5 := by
  sorry

end f_8_5_equals_1_5_l1237_123765


namespace amy_spending_at_fair_l1237_123756

/-- Amy's spending at the fair --/
theorem amy_spending_at_fair (initial_amount final_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : final_amount = 11) :
  initial_amount - final_amount = 4 := by
  sorry

end amy_spending_at_fair_l1237_123756


namespace remainder_of_645_l1237_123725

-- Define the set s
def s : Set ℕ := {n : ℕ | n > 0 ∧ ∃ k, n = 8 * k + 5}

-- Define the 81st element of s
def element_81 : ℕ := 645

-- Theorem statement
theorem remainder_of_645 : 
  element_81 ∈ s ∧ (∃ k : ℕ, element_81 = 8 * k + 5) :=
by sorry

end remainder_of_645_l1237_123725


namespace cube_root_plus_sqrt_minus_sqrt_l1237_123777

theorem cube_root_plus_sqrt_minus_sqrt : ∃ x y z : ℝ, x^3 = -64 ∧ y^2 = 9 ∧ z^2 = 25/16 ∧ x + y - z = -9/4 := by
  sorry

end cube_root_plus_sqrt_minus_sqrt_l1237_123777


namespace tangent_points_parallel_to_line_l1237_123737

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line :
  ∀ x y : ℝ, 
  (y = f x) ∧ 
  (f' x = 4) → 
  ((x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0)) :=
sorry

end tangent_points_parallel_to_line_l1237_123737


namespace amy_doll_cost_l1237_123789

def doll_cost (initial_amount : ℕ) (dolls_bought : ℕ) (remaining_amount : ℕ) : ℚ :=
  (initial_amount - remaining_amount : ℚ) / dolls_bought

theorem amy_doll_cost :
  doll_cost 100 3 97 = 1 := by
  sorry

end amy_doll_cost_l1237_123789


namespace jesses_friends_l1237_123728

theorem jesses_friends (bananas_per_friend : ℝ) (total_bananas : ℕ) 
  (h1 : bananas_per_friend = 21.0) 
  (h2 : total_bananas = 63) : 
  (total_bananas : ℝ) / bananas_per_friend = 3 := by
  sorry

end jesses_friends_l1237_123728


namespace gcd_1755_1242_l1237_123764

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := by
  sorry

end gcd_1755_1242_l1237_123764


namespace simplify_expression_l1237_123780

theorem simplify_expression (a b : ℝ) (hb : b ≠ 0) (ha : a ≠ b^(1/3)) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by sorry

end simplify_expression_l1237_123780


namespace parabola_focus_and_directrix_l1237_123708

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x - 2 = (y - 3)^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2.25, 3)

-- Define the directrix of the parabola
def directrix : ℝ → Prop := λ x => x = 1.75

theorem parabola_focus_and_directrix :
  ∀ x y : ℝ, parabola_equation x y →
  (∃ p : ℝ × ℝ, p = focus ∧ 
   ∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.1 - 1.75)^2) ∧
  (∀ q : ℝ × ℝ, parabola_equation q.1 q.2 → 
   ∃ r : ℝ, directrix r ∧ 
   (q.1 - focus.1)^2 + (q.2 - focus.2)^2 = (q.1 - r)^2) :=
by sorry


end parabola_focus_and_directrix_l1237_123708


namespace minimum_discount_l1237_123799

theorem minimum_discount (n : ℕ) : n = 38 ↔ 
  (n > 0) ∧
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^4 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.30) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^4 ∧
   (1 - n / 100 : ℝ) < (1 - 0.30) * (1 - 0.10)) :=
by sorry

end minimum_discount_l1237_123799


namespace f_difference_bound_l1237_123781

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 15

-- State the theorem
theorem f_difference_bound (x a : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end f_difference_bound_l1237_123781


namespace sin_690_degrees_l1237_123703

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by sorry

end sin_690_degrees_l1237_123703


namespace card_area_after_shortening_l1237_123768

/-- Given a rectangle with initial dimensions 5 × 7 inches, 
    prove that shortening both sides by 1 inch results in an area of 24 square inches. -/
theorem card_area_after_shortening :
  let initial_length : ℝ := 7
  let initial_width : ℝ := 5
  let shortened_length : ℝ := initial_length - 1
  let shortened_width : ℝ := initial_width - 1
  shortened_length * shortened_width = 24 := by
  sorry

end card_area_after_shortening_l1237_123768


namespace combined_cost_increase_percentage_l1237_123790

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem combined_cost_increase_percentage
  (bicycle_cost : ℝ)
  (helmet_cost : ℝ)
  (bicycle_increase_percent : ℝ)
  (helmet_increase_percent : ℝ)
  (h1 : bicycle_cost = 160)
  (h2 : helmet_cost = 40)
  (h3 : bicycle_increase_percent = 5)
  (h4 : helmet_increase_percent = 10) :
  let new_bicycle_cost := bicycle_cost * (1 + bicycle_increase_percent / 100)
  let new_helmet_cost := helmet_cost * (1 + helmet_increase_percent / 100)
  let original_total := bicycle_cost + helmet_cost
  let new_total := new_bicycle_cost + new_helmet_cost
  (new_total - original_total) / original_total * 100 = 6 := by
  sorry

#check combined_cost_increase_percentage

end combined_cost_increase_percentage_l1237_123790


namespace solve_equation_l1237_123775

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.01) : x = 0.9 := by
  sorry

end solve_equation_l1237_123775


namespace permutation_equation_solution_l1237_123724

def A (n : ℕ) : ℕ := n * (n - 1)

theorem permutation_equation_solution :
  ∃ (x : ℕ), 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2 ∧ x = 4 := by
  sorry

end permutation_equation_solution_l1237_123724


namespace salary_changes_l1237_123779

/-- Represents the series of salary changes and calculates the final salary --/
def final_salary (original : ℝ) : ℝ :=
  let after_first_raise := original * 1.12
  let after_reduction := after_first_raise * 0.93
  let after_bonus := after_reduction * 1.15
  let fixed_component := after_bonus * 0.7
  let variable_component := after_bonus * 0.3 * 0.9
  fixed_component + variable_component

/-- Theorem stating that an original salary of approximately 7041.77 results in a final salary of 7600.35 --/
theorem salary_changes (ε : ℝ) (hε : ε > 0) :
  ∃ (original : ℝ), abs (original - 7041.77) < ε ∧ final_salary original = 7600.35 := by
  sorry

end salary_changes_l1237_123779


namespace distribute_seven_balls_four_boxes_l1237_123738

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 890 ways to distribute 7 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 890 := by
  sorry

end distribute_seven_balls_four_boxes_l1237_123738


namespace shopkeeper_cards_l1237_123783

/-- The number of cards in a complete deck of standard playing cards -/
def standard_deck : ℕ := 52

/-- The number of cards in a complete deck of Uno cards -/
def uno_deck : ℕ := 108

/-- The number of cards in a complete deck of tarot cards -/
def tarot_deck : ℕ := 78

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 4

/-- The number of complete decks of Uno cards -/
def uno_decks : ℕ := 3

/-- The number of complete decks of tarot cards -/
def tarot_decks : ℕ := 5

/-- The number of additional standard playing cards -/
def extra_standard : ℕ := 12

/-- The number of additional Uno cards -/
def extra_uno : ℕ := 7

/-- The number of additional tarot cards -/
def extra_tarot : ℕ := 9

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * standard_deck + extra_standard +
  uno_decks * uno_deck + extra_uno +
  tarot_decks * tarot_deck + extra_tarot

theorem shopkeeper_cards : total_cards = 950 := by
  sorry

end shopkeeper_cards_l1237_123783


namespace first_student_stickers_l1237_123705

/-- Given a sequence of gold sticker counts for students 2 to 6, 
    prove that the first student received 29 stickers. -/
theorem first_student_stickers 
  (second : ℕ) 
  (third : ℕ) 
  (fourth : ℕ) 
  (fifth : ℕ) 
  (sixth : ℕ) 
  (h1 : second = 35) 
  (h2 : third = 41) 
  (h3 : fourth = 47) 
  (h4 : fifth = 53) 
  (h5 : sixth = 59) : 
  second - 6 = 29 := by
  sorry

end first_student_stickers_l1237_123705


namespace max_bishops_on_8x8_chessboard_l1237_123796

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (is_valid : size = 8)

/-- Represents a bishop placement on the chessboard --/
structure BishopPlacement :=
  (board : Chessboard)
  (num_bishops : Nat)
  (max_per_diagonal : Nat)
  (is_valid : max_per_diagonal = 3)

/-- The maximum number of bishops that can be placed on the chessboard --/
def max_bishops (placement : BishopPlacement) : Nat :=
  38

/-- Theorem stating the maximum number of bishops on an 8x8 chessboard --/
theorem max_bishops_on_8x8_chessboard (placement : BishopPlacement) :
  placement.board.size = 8 →
  placement.max_per_diagonal = 3 →
  max_bishops placement = 38 := by
  sorry

end max_bishops_on_8x8_chessboard_l1237_123796


namespace sum_of_roots_eq_one_l1237_123739

theorem sum_of_roots_eq_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 3) * (x₁ - 4) = 22 ∧ 
                 (x₂ + 3) * (x₂ - 4) = 22 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_eq_one_l1237_123739


namespace exists_divisible_by_33_l1237_123758

def original_number : ℕ := 975312468

def insert_digit (n : ℕ) (d : ℕ) (pos : ℕ) : ℕ :=
  let digits := n.digits 10
  let (before, after) := digits.splitAt pos
  ((before ++ [d] ++ after).foldl (fun acc x => acc * 10 + x) 0)

theorem exists_divisible_by_33 :
  ∃ (d : ℕ) (pos : ℕ), d < 10 ∧ pos ≤ 9 ∧ 
  (insert_digit original_number d pos) % 33 = 0 :=
sorry

end exists_divisible_by_33_l1237_123758


namespace symmetric_points_line_intercept_l1237_123714

/-- Given two points A and B symmetric with respect to a line y = kx + b,
    prove that the x-intercept of the line is 5/6. -/
theorem symmetric_points_line_intercept 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (-2, 1)) 
  (k b : ℝ) 
  (h_symmetric : B = (2 * ((k * A.1 + b) / (1 + k^2) - k * A.2 / (1 + k^2)) - A.1,
                      2 * (k * (k * A.1 + b) / (1 + k^2) + A.2 / (1 + k^2)) - A.2)) :
  (- b / k : ℝ) = 5/6 := by
  sorry

end symmetric_points_line_intercept_l1237_123714


namespace factorization_y_squared_minus_one_l1237_123723

/-- A factorization is valid if the expanded form equals the factored form -/
def IsValidFactorization (expanded factored : ℝ → ℝ) : Prop :=
  ∀ x, expanded x = factored x

/-- A factorization is from left to right if it's in the form of factors multiplied together -/
def IsFactorizationLeftToRight (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x * h x

theorem factorization_y_squared_minus_one :
  IsValidFactorization (fun y => y^2 - 1) (fun y => (y + 1) * (y - 1)) ∧
  IsFactorizationLeftToRight (fun y => (y + 1) * (y - 1)) ∧
  ¬IsValidFactorization (fun x => x * (a - b)) (fun x => a*x - b*x) ∧
  ¬IsValidFactorization (fun x => x^2 - 2*x) (fun x => x * (x - 2/x)) ∧
  ¬IsFactorizationLeftToRight (fun x => x * (a + b) + c) :=
by sorry

end factorization_y_squared_minus_one_l1237_123723


namespace open_box_volume_l1237_123769

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 4) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4480 := by
  sorry

end open_box_volume_l1237_123769


namespace linear_function_shift_l1237_123706

/-- Given a linear function y = mx - 1, prove that if the graph is shifted down by 2 units
    and passes through the point (-2, 1), then m = -2. -/
theorem linear_function_shift (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 3 → (x = -2 ∧ y = 1)) → m = -2 := by
  sorry

end linear_function_shift_l1237_123706


namespace circle_distance_range_l1237_123707

theorem circle_distance_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  3 - 2 * Real.sqrt 2 ≤ x^2 - 2*x + y^2 + 2*y + 2 ∧ 
  x^2 - 2*x + y^2 + 2*y + 2 ≤ 3 + 2 * Real.sqrt 2 := by
  sorry

end circle_distance_range_l1237_123707


namespace smallest_number_of_students_l1237_123709

theorem smallest_number_of_students (grade12 grade11 grade10 : ℕ) : 
  grade12 > 0 ∧ grade11 > 0 ∧ grade10 > 0 →
  grade12 * 3 = grade10 * 4 →
  grade12 * 5 = grade11 * 7 →
  grade11 * 9 = grade10 * 10 →
  grade12 + grade11 + grade10 ≥ 66 :=
by
  sorry

end smallest_number_of_students_l1237_123709


namespace intersection_A_B_range_of_m_l1237_123736

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m) / (x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x ≤ 1/3) ∨ (1/2 ≤ x ∧ x < 6)} := by sorry

-- Theorem for the range of m when A ∪ C = C
theorem range_of_m (m : ℝ) :
  (A ∪ C m = C m) → (-3 ≤ m ∧ m ≤ -1) := by sorry

end intersection_A_B_range_of_m_l1237_123736


namespace linear_function_properties_l1237_123757

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the original line before moving up
def g (x : ℝ) : ℝ := 2 * x - 4

-- Define the line after moving up by 5 units
def h (x : ℝ) : ℝ := g x + 5

theorem linear_function_properties :
  (∀ x y : ℝ, x < 0 ∧ y < 0 → f x ≠ y) ∧
  (∀ x : ℝ, h x = 2 * x + 1) := by
  sorry

end linear_function_properties_l1237_123757


namespace simplify_complex_fraction_l1237_123743

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) + (2 / (Real.sqrt 7 - 2)))) =
  ((6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35)) := by
  sorry

end simplify_complex_fraction_l1237_123743


namespace inequality_proof_l1237_123742

theorem inequality_proof (m n : ℕ+) : 
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 := by
  sorry

end inequality_proof_l1237_123742


namespace license_plate_count_l1237_123755

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 4

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : total_license_plates = 878800000 := by
  sorry

end license_plate_count_l1237_123755


namespace golf_ball_distribution_returns_to_initial_state_l1237_123721

/-- Represents the distribution of golf balls in boxes and the next starting box -/
structure State :=
  (balls : Fin 10 → ℕ)
  (next_box : Fin 10)

/-- The set of all possible states -/
def S : Set State := {s | ∀ i, s.balls i > 0}

/-- Represents one move in the game -/
def move (s : State) : State :=
  sorry

theorem golf_ball_distribution_returns_to_initial_state :
  ∀ (initial : State),
  initial ∈ S →
  ∃ (n : ℕ+),
  (move^[n] initial) = initial :=
sorry

end golf_ball_distribution_returns_to_initial_state_l1237_123721


namespace third_pile_balls_l1237_123749

theorem third_pile_balls (a b c : ℕ) (x : ℕ) :
  a + b + c = 2012 →
  b - x = 17 →
  a - x = 2 * (c - x) →
  c = 665 := by
sorry

end third_pile_balls_l1237_123749


namespace hyperbola_eccentricity_l1237_123726

/-- A hyperbola with foci on the x-axis and asymptotic lines y = ±√3x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = Real.sqrt 3) → 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 := by sorry

end hyperbola_eccentricity_l1237_123726


namespace inverse_direct_variation_l1237_123782

/-- Given that a²c varies inversely with b³ and c varies directly as b², 
    prove that a² = 25/128 when b = 4, given initial conditions. -/
theorem inverse_direct_variation (a b c : ℝ) (k k' : ℝ) : 
  (∀ a b c, a^2 * c * b^3 = k) →  -- a²c varies inversely with b³
  (∀ b c, c = k' * b^2) →         -- c varies directly as b²
  (5^2 * 12 * 2^3 = k) →          -- initial condition for k
  (12 = k' * 2^2) →               -- initial condition for k'
  (∀ a, a^2 * (k' * 4^2) * 4^3 = k) →  -- condition for b = 4
  (∃ a, a^2 = 25 / 128) :=
by sorry

end inverse_direct_variation_l1237_123782


namespace f_composition_value_l1237_123760

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x + 2 * Real.cos (2 * x) else -Real.exp (2 * x)

theorem f_composition_value : f (f (Real.pi / 2)) = -1 / Real.exp 2 := by
  sorry

end f_composition_value_l1237_123760


namespace sector_area_l1237_123734

/-- Given a sector with central angle θ and arc length L, 
    the area A of the sector can be calculated. -/
theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = 2 → L = 4 → A = 4 → A = (1/2) * (L/θ)^2 * θ :=
by sorry

end sector_area_l1237_123734


namespace mikes_pens_l1237_123700

theorem mikes_pens (initial_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 5 → final_pens = 40 → ∃ M : ℕ, 
    2 * (initial_pens + M) - 10 = final_pens ∧ M = 20 := by
  sorry

end mikes_pens_l1237_123700


namespace sphere_hemisphere_volume_ratio_l1237_123740

/-- The ratio of the volume of a sphere with radius p to the volume of a hemisphere with radius 3p is 1/13.5 -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 1 / 13.5 :=
by sorry

end sphere_hemisphere_volume_ratio_l1237_123740


namespace candy_division_l1237_123718

theorem candy_division (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 42 →
  num_bags = 2 →
  candy_per_bag * num_bags = total_candy →
  candy_per_bag = 21 := by
  sorry

end candy_division_l1237_123718


namespace mlb_game_ratio_l1237_123748

theorem mlb_game_ratio (misses : ℕ) (total : ℕ) : 
  misses = 50 → total = 200 → (misses : ℚ) / (total - misses : ℚ) = 1 / 3 := by
  sorry

end mlb_game_ratio_l1237_123748


namespace geometric_sequence_first_term_l1237_123722

/-- Given a geometric sequence where the fourth term is 32 and the fifth term is 64, prove that the first term is 4. -/
theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 32 = c * r ∧ 64 = 32 * r) → a = 4 := by
  sorry

end geometric_sequence_first_term_l1237_123722


namespace south_five_is_negative_five_l1237_123747

/-- Represents the direction of movement -/
inductive Direction
| North
| South

/-- Represents a movement with magnitude and direction -/
structure Movement where
  magnitude : ℕ
  direction : Direction

/-- Function to convert a movement to its signed representation -/
def movementToSigned (m : Movement) : ℤ :=
  match m.direction with
  | Direction.North => m.magnitude
  | Direction.South => -m.magnitude

theorem south_five_is_negative_five :
  let southFive : Movement := ⟨5, Direction.South⟩
  movementToSigned southFive = -5 := by sorry

end south_five_is_negative_five_l1237_123747


namespace flu_transmission_rate_l1237_123712

theorem flu_transmission_rate : 
  ∃ x : ℝ, 
    x > 0 ∧ 
    (1 + x) + x * (1 + x) = 100 ∧ 
    x = 9 := by
  sorry

end flu_transmission_rate_l1237_123712
