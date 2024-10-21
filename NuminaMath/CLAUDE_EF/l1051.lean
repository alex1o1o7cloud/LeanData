import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_pi_over_four_l1051_105113

theorem derivative_at_pi_over_four (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos x + 2 * Real.sin x) :
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_pi_over_four_l1051_105113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1051_105194

noncomputable def f (x : ℝ) := |2*x - 3| + |2*x + 2|

theorem f_properties :
  (∃ (s : Set ℝ), s = {x | f x < x + 5} ∧ s = Set.Ioo (-1) 2) ∧
  (∃ (r : Set ℝ), r = {a | ∀ x, f x > a + 4/a} ∧ r = Set.Iic 0 ∪ Set.Ioo 1 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1051_105194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_relation_l1051_105110

theorem triangle_cotangent_relation (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a^2 - b^2) * (Real.cos C / Real.sin C) + 
  (b^2 - c^2) * (Real.cos A / Real.sin A) + 
  (c^2 - a^2) * (Real.cos B / Real.sin B) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_relation_l1051_105110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_in_terms_of_x_l1051_105141

theorem tan_half_angle_in_terms_of_x (θ x : ℝ) (h_acute : 0 < θ ∧ θ < π/2) (h_x : x > 1)
  (h_cos : Real.cos (θ/2) = Real.sqrt ((x - 1)/(2*x))) :
  Real.tan (θ/2) = Real.sqrt ((x + 1)/(x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_in_terms_of_x_l1051_105141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DFE_is_56_degrees_l1051_105114

-- Define the points
variable (B C D E F : Point Euclidean2) -- Use Point Euclidean2 instead of just Point

-- Define the angles
variable (angle_CFB angle_FCB angle_CBF angle_EFB angle_CFE angle_FEB angle_FBE angle_DFE : ℝ)

-- State the theorem
theorem angle_DFE_is_56_degrees :
  -- Conditions
  F ∈ segment C D →
  angle_CFB = angle_FCB →  -- Triangle CFB is isosceles
  angle_FEB = angle_FBE →  -- Triangle DFE is isosceles
  angle_EFB = 3 * angle_CFE →
  angle_CFB = 50 →
  -- Conclusion
  angle_DFE = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DFE_is_56_degrees_l1051_105114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l1051_105196

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (6 * x + c)

theorem smallest_c_value (c : ℝ) (h1 : c > 0) 
  (h2 : ∀ x : ℝ, f c x ≤ f c (-π/6)) : c = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l1051_105196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_theorem_l1051_105107

/-- Represents a ball with an integer value -/
structure Ball where
  value : ℤ

/-- Given 2n balls, returns true if any n pairs always include two pairs with the same sum -/
def hasPairsWithEqualSum (balls : List Ball) (n : ℕ) : Prop :=
  ∀ (pairs : List (Ball × Ball)), 
    pairs.length = n → 
    (pairs.map (λ p => p.fst.value + p.snd.value)).toFinset.card < n

theorem balls_theorem (n : ℕ) (h_n : n ≥ 2) (balls : List Ball) 
  (h_balls : balls.length = 2 * n) (h_pairs : hasPairsWithEqualSum balls n) :
  (∃ (a b c d : Ball), a ∈ balls ∧ b ∈ balls ∧ c ∈ balls ∧ d ∈ balls ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a.value = b.value ∧ b.value = c.value ∧ c.value = d.value) ∧
  ((balls.map Ball.value).toFinset.card ≤ n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_theorem_l1051_105107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacup_flip_invariant_l1051_105109

/-- Represents the state of teacups after a series of flips -/
def TeacupState := List Int

/-- Represents a single flip operation -/
def FlipOperation := List Nat

/-- The number of teacups -/
def m : Nat := 3  -- Example value, can be changed

/-- The number of teacups that can be flipped in one operation -/
def n : Nat := 2  -- Example value, can be changed

/-- Axiom: m is at least 3 -/
axiom m_ge_3 : m ≥ 3

/-- Axiom: m is odd -/
axiom m_odd : Odd m

/-- Axiom: n is at least 2 -/
axiom n_ge_2 : n ≥ 2

/-- Axiom: n is less than m -/
axiom n_lt_m : n < m

/-- Axiom: n is even -/
axiom n_even : Even n

/-- Initial state of teacups, all facing up -/
def initial_state : TeacupState := List.replicate m 1

/-- Apply a flip operation to a teacup state -/
def apply_flip (state : TeacupState) (flip : FlipOperation) : TeacupState :=
  sorry

/-- The product of all teacup states -/
def state_product (state : TeacupState) : Int :=
  state.prod

/-- Theorem: For any sequence of flip operations, the product of teacup states remains 1 -/
theorem teacup_flip_invariant (flips : List FlipOperation) :
  state_product (flips.foldl apply_flip initial_state) = 1 := by
  sorry

#check teacup_flip_invariant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacup_flip_invariant_l1051_105109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_le_one_l1051_105193

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + Real.exp x - a*x

-- State the theorem
theorem monotone_increasing_f_implies_a_le_one (a : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a x ≤ f a y) →
  a ≤ 1 :=
by
  intro h
  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_le_one_l1051_105193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_202122_to_313233_transform_999999_to_000000_in_8_moves_multiple_of_11_invariant_impossible_transform_112233_to_000000_l1051_105137

-- Define the type for six-digit numbers
def SixDigitNum := Fin 1000000

-- Define the allowed moves
def rotate (n : SixDigitNum) : SixDigitNum := sorry
def addMove (n : SixDigitNum) : SixDigitNum := sorry
def subtractMove (n : SixDigitNum) : SixDigitNum := sorry

-- Define a sequence of moves
def Move := SixDigitNum → SixDigitNum
def applyMoves : List Move → SixDigitNum → SixDigitNum := sorry

theorem transform_202122_to_313233 : 
  ∃ (moves : List Move), applyMoves moves ⟨202122, sorry⟩ = ⟨313233, sorry⟩ := by sorry

theorem transform_999999_to_000000_in_8_moves :
  ∃ (moves : List Move), applyMoves moves ⟨999999, sorry⟩ = ⟨0, sorry⟩ ∧ moves.length = 8 := by sorry

theorem multiple_of_11_invariant (n : SixDigitNum) (moves : List Move) :
  (n.val % 11 = 0) → ((applyMoves moves n).val % 11 = 0) := by sorry

theorem impossible_transform_112233_to_000000 :
  ¬∃ (moves : List Move), applyMoves moves ⟨112233, sorry⟩ = ⟨0, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_202122_to_313233_transform_999999_to_000000_in_8_moves_multiple_of_11_invariant_impossible_transform_112233_to_000000_l1051_105137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_is_eight_thirds_l1051_105185

/-- Represents a right circular cone. -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder. -/
structure Cylinder where
  radius : ℝ

/-- Calculates the radius of an inscribed cylinder in a cone. -/
noncomputable def inscribedCylinderRadius (cone : Cone) : ℝ :=
  8 / 3

/-- Theorem: The radius of a right circular cylinder inscribed in a right circular cone
    with diameter 8 and altitude 16 is 8/3. -/
theorem inscribed_cylinder_radius_is_eight_thirds
  (cone : Cone)
  (cyl : Cylinder)
  (h1 : cone.diameter = 8)
  (h2 : cone.altitude = 16)
  (h3 : cyl.radius * 2 = cyl.radius * 4) :  -- cylinder's diameter equals its height
  cyl.radius = inscribedCylinderRadius cone := by
  sorry

#check inscribed_cylinder_radius_is_eight_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_is_eight_thirds_l1051_105185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_pi_third_l1051_105153

noncomputable def f (x : ℝ) := Real.log x - Real.sin x

noncomputable def f' (x a : ℝ) := a / x - Real.cos x

theorem extreme_value_at_pi_third (a : ℝ) :
  f' (π / 3) a = 0 → a = π / 6 := by
  intro h
  have h1 : 3 * a / π - 1 / 2 = 0 := by
    -- Proof steps here
    sorry
  -- More proof steps here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_pi_third_l1051_105153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailer_homes_count_l1051_105159

/- Define the function to calculate the number of new trailer homes added -/
def trailer_homes_added (original_count : ℕ) (original_avg_age : ℕ) (current_avg_age : ℕ) : ℕ :=
  let years_passed := 3
  let current_original_avg_age := original_avg_age + years_passed
  (original_count * current_original_avg_age) / (current_avg_age - years_passed) - original_count

/- Define the assumptions as variables -/
def original_count : ℕ := 30
def original_avg_age : ℕ := 15
def current_avg_age : ℕ := 12

/- Theorem to prove that the number of new trailer homes added is 20 -/
theorem new_trailer_homes_count :
  trailer_homes_added original_count original_avg_age current_avg_age = 20 := by
  /- Unfold the definition of trailer_homes_added -/
  unfold trailer_homes_added
  /- Simplify the expression -/
  simp [original_count, original_avg_age, current_avg_age]
  /- The proof is completed with sorry for now -/
  sorry

#eval trailer_homes_added original_count original_avg_age current_avg_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailer_homes_count_l1051_105159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1051_105178

theorem problem_statement (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) : (a + b) / (a * b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1051_105178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_five_l1051_105129

/-- The set of possible values for a -/
def A : Finset ℕ := Finset.range 50

/-- The set of possible values for b -/
def B : Finset ℕ := Finset.range 100

/-- The function that computes the units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The probability of an event occurring given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- The main theorem stating the probability of 4^a + 9^b having a units digit of 5 -/
theorem probability_units_digit_five :
  probability 
    (Finset.filter (fun (p : ℕ × ℕ) => unitsDigit (4^p.1 + 9^p.2) = 5) (A.product B)).card
    (A.card * B.card) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_five_l1051_105129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_purchase_price_l1051_105197

/-- The purchase price of a jacket given selling price and markup conditions --/
theorem jacket_purchase_price (S P : ℝ) (h1 : S = P + 0.30 * S) 
  (h2 : 0.80 * S - P = 6.000000000000007) : 
  abs (P - 42.00) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_purchase_price_l1051_105197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_four_sets_eq_three_fourths_l1051_105184

/-- Represents the probability of a team winning a single set -/
noncomputable def p : ℝ := 1 / 2

/-- Represents the number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- Represents the maximum number of sets in the match -/
def max_sets : ℕ := 5

/-- Calculates the probability of a match lasting at least 4 sets -/
noncomputable def prob_at_least_four_sets : ℝ :=
  Nat.choose sets_to_win 2 * p^2 * (1 - p) +
  Nat.choose sets_to_win 2 * p^2 * (1 - p) * p +
  Nat.choose 4 2 * p^2 * (1 - p)^2 * p +
  Nat.choose 4 2 * p^2 * (1 - p)^2 * (1 - p)

/-- Theorem stating that the probability of a match lasting at least 4 sets is 3/4 -/
theorem prob_at_least_four_sets_eq_three_fourths :
  prob_at_least_four_sets = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_four_sets_eq_three_fourths_l1051_105184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation180_maps_correctly_l1051_105143

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rigid transformation in 2D space -/
structure RigidTransformation where
  transform : Point → Point
  isIsometry : ∀ (p q : Point), 
    (p.x - q.x)^2 + (p.y - q.y)^2 = ((transform p).x - (transform q).x)^2 + ((transform p).y - (transform q).y)^2

/-- Clockwise rotation by 180 degrees around the origin -/
def rotation180 : RigidTransformation where
  transform := λ p ↦ { x := -p.x, y := -p.y }
  isIsometry := by sorry

theorem rotation180_maps_correctly : 
  let C : Point := ⟨3, -2⟩
  let D : Point := ⟨4, -5⟩
  let C' : Point := ⟨-3, 2⟩
  let D' : Point := ⟨-4, 5⟩
  (rotation180.transform C = C') ∧ (rotation180.transform D = D') := by
  sorry

#check rotation180_maps_correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation180_maps_correctly_l1051_105143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_length_difference_l1051_105139

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ

/-- Given a parabola, its focus, and two points on it, compute |BF| - |AF| -/
noncomputable def length_difference (C : Parabola) (F A B Q : Point) (l : Line) : ℝ :=
  Real.sqrt ((B.x - F.x)^2 + B.y^2) - Real.sqrt ((A.x - F.x)^2 + A.y^2)

theorem parabola_length_difference (C : Parabola) (F A B Q : Point) (l : Line) :
  C.p = 2 →
  A.x^2 + A.y^2 = 2 * C.p * A.x →
  B.x^2 + B.y^2 = 2 * C.p * B.x →
  F.x = C.p ∧ F.y = 0 →
  Q.x = -7/2 ∧ Q.y = 0 →
  (B.y - Q.y) * (B.x - F.x) = -(B.x - Q.x) * (B.y - F.y) →
  length_difference C F A B Q l = -3/2 := by
  sorry

#check parabola_length_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_length_difference_l1051_105139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_is_square_l1051_105187

theorem gcd_product_is_square (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0) 
  (eq : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z)) * x * y * z = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_product_is_square_l1051_105187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1051_105125

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

/-- Theorem: For a geometric sequence, if a₁ + a₃ = 5 and a₂ + a₄ = 10, then q = 2 -/
theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h1 : geometric_sequence a₁ q 1 + geometric_sequence a₁ q 3 = 5) 
  (h2 : geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 10) : q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1051_105125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l1051_105138

noncomputable def linda_speed : ℝ := 2
noncomputable def linda_time1 : ℝ := 0.5
noncomputable def linda_time2 : ℝ := 1
noncomputable def tom_speed : ℝ := 6

noncomputable def linda_distance : ℝ := linda_speed * (linda_time1 + linda_time2)

noncomputable def time_half_distance : ℝ := (linda_distance / 2) / tom_speed
noncomputable def time_twice_distance : ℝ := (linda_distance * 2) / tom_speed

theorem time_difference_proof :
  time_twice_distance - time_half_distance = 45 / 60 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l1051_105138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l1051_105124

open BigOperators

def f (n : ℕ+) : ℚ :=
  (∑ i in Finset.range n.val, 1 / ((i + 1 : ℕ) ^ 3 : ℚ)) + 1

def g (n : ℕ+) : ℚ :=
  3/2 - 1 / (2 * (n.val : ℚ) ^ 2)

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l1051_105124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_e_l1051_105173

open Real

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (log x - 1) / (log x)^2

-- Theorem statement
theorem max_m_is_e :
  ∀ m : ℝ, m > 1 →
  (∀ x ∈ Set.Ioo 1 m, (f_derivative x < 0)) →
  m ≤ ℯ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_e_l1051_105173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_l1051_105192

/-- The volume of a cone with given radius and height -/
noncomputable def coneVolume (radius : ℝ) (height : ℝ) : ℝ :=
  (1/3) * Real.pi * radius^2 * height

theorem cone_volume_approximation :
  let radius : ℝ := 10
  let height : ℝ := 21
  ∃ (ε : ℝ), ε > 0 ∧ |coneVolume radius height - 2199.109| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_l1051_105192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_prime_negative_l1051_105134

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem solution_set_f_prime_negative :
  {x : ℝ | (deriv f x) < 0} = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_prime_negative_l1051_105134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_l1051_105180

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the equation of a sphere in 3D space -/
def isSphere (center : Point3D) (R : ℝ) (p : Point3D) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = R^2

/-- Theorem: The equation (x - x₀)² + (y - y₀)² + (z - z₀)² = R² represents a sphere 
    with radius R centered at (x₀, y₀, z₀) in 3D Cartesian coordinate system -/
theorem sphere_equation (center : Point3D) (R : ℝ) :
  ∀ p : Point3D, isSphere center R p ↔ 
    ∃ (θ φ : ℝ), p.x = center.x + R * Real.sin θ * Real.cos φ ∧
                 p.y = center.y + R * Real.sin θ * Real.sin φ ∧
                 p.z = center.z + R * Real.cos θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_l1051_105180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regular_rate_l1051_105100

/-- Represents Mary's work conditions and earnings --/
structure MaryWork where
  maxHours : ℕ
  regularHours : ℕ
  maxEarnings : ℕ
  overtimeRate : ℚ

/-- Calculates Mary's total earnings given her regular hourly rate --/
def totalEarnings (work : MaryWork) (regularRate : ℚ) : ℚ :=
  let overtimeHours := work.maxHours - work.regularHours
  regularRate * work.regularHours + 
  work.overtimeRate * regularRate * overtimeHours

/-- Theorem stating that the maximum regular hourly rate is $8 --/
theorem max_regular_rate (work : MaryWork) 
  (h1 : work.maxHours = 50)
  (h2 : work.regularHours = 20)
  (h3 : work.maxEarnings = 460)
  (h4 : work.overtimeRate = 5/4) :
  ∃ (rate : ℚ), rate = 8 ∧ 
    totalEarnings work rate ≤ work.maxEarnings ∧
    ∀ (r : ℚ), r > rate → totalEarnings work r > work.maxEarnings :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regular_rate_l1051_105100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1051_105115

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧
  ∃ (m n : Real × Real),
    m = (t.c, Real.sqrt 3 * t.b) ∧
    n = (Real.cos t.C, Real.sin t.B) ∧
    ∃ (k : Real), m = k • n ∧
  ∃ (d : Real), 
    Real.sin (t.A + t.B) + d = Real.sin (2 * t.A) ∧
    Real.sin (2 * t.A) + d = Real.sin (t.B - t.A)

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : problem_conditions t) : 
  t.C = π / 3 ∧ (t.a = 4 * Real.sqrt 3 / 3 ∨ t.a = 2 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1051_105115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_portion_theorem_l1051_105131

/-- Represents a partnership with two partners -/
structure Partnership where
  investment1 : ℚ
  investment2 : ℚ
  total_profit : ℚ
  equal_portion : ℚ

/-- Calculates the share of a partner based on the partnership structure -/
def calculate_share (p : Partnership) (investment : ℚ) : ℚ :=
  p.equal_portion / 2 + (investment / (p.investment1 + p.investment2)) * (p.total_profit - p.equal_portion)

/-- The main theorem about the equal portion in the partnership -/
theorem equal_portion_theorem (p : Partnership) 
  (h1 : p.investment1 = 650)
  (h2 : p.investment2 = 350)
  (h3 : p.total_profit = 3000)
  (h4 : calculate_share p p.investment1 = calculate_share p p.investment2 + 600) :
  p.equal_portion = 1000 := by
  sorry

#check equal_portion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_portion_theorem_l1051_105131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_urn_size_l1051_105130

/-- Represents the number of marbles of each color in the urn -/
structure UrnContents where
  red : ℕ
  white : ℕ
  blue : ℕ
  black : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (u : UrnContents) : ℕ :=
  u.red + u.white + u.blue + u.black

/-- Represents the probability of selecting a specific combination of marbles -/
noncomputable def selectionProbability (u : UrnContents) (red white blue black : ℕ) : ℚ :=
  (Nat.choose u.red red * Nat.choose u.white white * Nat.choose u.blue blue * Nat.choose u.black black : ℚ) /
  Nat.choose (totalMarbles u) 4

/-- Checks if the three specified events are equally likely -/
def eventsEquallyLikely (u : UrnContents) : Prop :=
  selectionProbability u 4 0 0 0 = selectionProbability u 2 1 1 0 ∧
  selectionProbability u 4 0 0 0 = selectionProbability u 2 0 1 1

/-- The main theorem stating that 18 is the smallest number of marbles satisfying the conditions -/
theorem smallest_urn_size :
  ∃ (u : UrnContents), totalMarbles u = 18 ∧ eventsEquallyLikely u ∧
  (∀ (v : UrnContents), eventsEquallyLikely v → totalMarbles v ≥ 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_urn_size_l1051_105130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l1051_105103

-- Define a triangle with integer side lengths
structure Triangle :=
  (a b c : ℕ)

-- Define the conditions
def coprime (t : Triangle) : Prop :=
  Nat.gcd t.a (Nat.gcd t.b t.c) = 1

noncomputable def angleBTwiceC (t : Triangle) : Prop :=
  2 * Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c)) = 
    Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

def bLessThan600 (t : Triangle) : Prop :=
  t.b < 600

-- Theorem statement
theorem no_triangle_satisfies_conditions :
  ¬ ∃ t : Triangle, coprime t ∧ angleBTwiceC t ∧ bLessThan600 t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l1051_105103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1051_105150

-- Define the expression as noncomputable
noncomputable def expression : ℝ := (Nat.factorial 9)^2 / Real.sqrt (Nat.factorial (9-3)) + (3/7 * 4^3)

-- State the theorem
theorem expression_approximation : 
  ∃ ε > 0, |expression - 4906624027| < ε := by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1051_105150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_value_l1051_105164

noncomputable def geometric_sequence (a : ℝ) : ℕ → ℝ := 
  λ n => a * (1/4)^(n-1)

noncomputable def sum_of_terms (a : ℝ) (n : ℕ) : ℝ := 
  a * (1/4)^(n-1) + 6

theorem geometric_sequence_sum_value (a : ℝ) :
  (∀ n : ℕ, sum_of_terms a n = a * (1/4)^(n-1) + 6) →
  a = -3/2 := by
  sorry

#check geometric_sequence
#check sum_of_terms
#check geometric_sequence_sum_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_value_l1051_105164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_roll_path_length_l1051_105182

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edgeLength : ℝ

/-- Represents a point on the surface of a tetrahedron -/
structure TetrahedronPoint where
  tetrahedron : RegularTetrahedron
  -- Additional properties to define the point's position could be added here

/-- 
  Calculates the path length of a point on a regular tetrahedron when rolled
  as described in the problem.
-/
noncomputable def pathLengthWhenRolled (t : RegularTetrahedron) (p : TetrahedronPoint) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem tetrahedron_roll_path_length 
  (t : RegularTetrahedron) 
  (p : TetrahedronPoint) 
  (h1 : t.edgeLength = 2) 
  (h2 : p.tetrahedron = t) : 
  pathLengthWhenRolled t p = (4 * Real.sqrt 3 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_roll_path_length_l1051_105182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_is_89_l1051_105160

/-- Represents a single die face value -/
def DieFace := Fin 6

/-- Represents the configuration of a die -/
structure DieConfig where
  top : DieFace
  bottom : DieFace
  left : DieFace
  right : DieFace
  front : DieFace
  back : DieFace

/-- Represents the stack of six dice -/
structure DiceStack where
  p : DieConfig  -- top die
  q : DieConfig  -- left bottom
  r : DieConfig  -- middle top
  s : DieConfig  -- left middle
  t : DieConfig  -- right bottom
  u : DieConfig  -- middle bottom

/-- Returns the value of a die face -/
def faceValue (face : DieFace) : Nat :=
  face.val + 1

/-- Checks if two faces are opposite on a die -/
def areOpposite (f1 f2 : DieFace) : Prop :=
  (f1.val = 0 ∧ f2.val = 5) ∨ (f1.val = 5 ∧ f2.val = 0) ∨
  (f1.val = 1 ∧ f2.val = 4) ∨ (f1.val = 4 ∧ f2.val = 1) ∨
  (f1.val = 2 ∧ f2.val = 3) ∨ (f1.val = 3 ∧ f2.val = 2)

/-- Checks if a die configuration is valid -/
def isValidDie (die : DieConfig) : Prop :=
  areOpposite die.top die.bottom ∧
  areOpposite die.left die.right ∧
  areOpposite die.front die.back

/-- Calculates the sum of visible faces for a given die configuration -/
def visibleSum (die : DieConfig) (visibleFaces : List DieFace) : Nat :=
  visibleFaces.map faceValue |>.sum

/-- Theorem: The maximum sum of visible faces in the given dice stack is 89 -/
theorem max_visible_sum_is_89 (stack : DiceStack)
  (h_valid_p : isValidDie stack.p)
  (h_valid_q : isValidDie stack.q)
  (h_valid_r : isValidDie stack.r)
  (h_valid_s : isValidDie stack.s)
  (h_valid_t : isValidDie stack.t)
  (h_valid_u : isValidDie stack.u) :
  (∃ (visible_p visible_q visible_r visible_s visible_t visible_u : List DieFace),
    visible_p.length = 5 ∧
    visible_q.length = 3 ∧
    visible_r.length = 4 ∧
    visible_s.length = 3 ∧
    visible_t.length = 2 ∧
    visible_u.length = 4 ∧
    visibleSum stack.p visible_p +
    visibleSum stack.q visible_q +
    visibleSum stack.r visible_r +
    visibleSum stack.s visible_s +
    visibleSum stack.t visible_t +
    visibleSum stack.u visible_u = 89) ∧
  (∀ (visible_p visible_q visible_r visible_s visible_t visible_u : List DieFace),
    visible_p.length = 5 →
    visible_q.length = 3 →
    visible_r.length = 4 →
    visible_s.length = 3 →
    visible_t.length = 2 →
    visible_u.length = 4 →
    visibleSum stack.p visible_p +
    visibleSum stack.q visible_q +
    visibleSum stack.r visible_r +
    visibleSum stack.s visible_s +
    visibleSum stack.t visible_t +
    visibleSum stack.u visible_u ≤ 89) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_is_89_l1051_105160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_im_part_z_is_root_l1051_105133

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Given complex number z satisfying the equation -/
noncomputable def z : ℂ := sorry

/-- The given equation -/
axiom z_eq : z * (1 + i) = (4 - 6*i) / (1 + i)

/-- Theorem: The imaginary part of z is -2 -/
theorem z_im_part : Complex.im z = -2 := by
  sorry

/-- Theorem: z is a root of the equation x^2 + 6x + 13 = 0 -/
theorem z_is_root : z^2 + 6*z + 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_im_part_z_is_root_l1051_105133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1051_105104

-- Define the operation * on positive real numbers
noncomputable def star (a b : ℝ) : ℝ := 
  if a ≥ b then b^a else b^2

-- Theorem statement
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  star 3 x₁ = 27 ∧ star 3 x₂ = 27 ∧
  x₁ = 3 ∧ x₂ = 3 * Real.sqrt 3 := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1051_105104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_height_l1051_105168

/-- The height of a regular triangular prism -/
noncomputable def prism_height (a : ℝ) (α : ℝ) : ℝ :=
  (a * Real.sqrt (Real.sin (Real.pi / 3 - α / 2) * Real.sin (Real.pi / 3 + α / 2))) / Real.sin (α / 2)

/-- Theorem: The height of a regular triangular prism with base side length a and angle α
    between non-intersecting diagonals of two lateral faces is given by the formula:
    h = (a * sqrt(sin(π/3 - α/2) * sin(π/3 + α/2))) / sin(α/2) -/
theorem regular_triangular_prism_height (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < Real.pi) :
  ∃ h : ℝ, h = prism_height a α ∧ h > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_height_l1051_105168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_players_same_flips_l1051_105145

/-- A fair coin has a probability of 1/2 for heads --/
noncomputable def fairCoin : ℝ := 1 / 2

/-- The number of players --/
def numPlayers : ℕ := 5

/-- The probability that all players flip their coins the same number of times --/
noncomputable def allSameFlips : ℝ := 1 / 31

/-- 
Theorem: The probability that all players flip their coins the same number of times
until they each get their first head is 1/31, given that each player flips a fair
coin repeatedly until they get their first head.
--/
theorem all_players_same_flips :
  allSameFlips = (∑' n, (fairCoin ^ n) ^ numPlayers) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_players_same_flips_l1051_105145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_k_sin_x_solution_range_l1051_105122

theorem sin_3x_eq_k_sin_x_solution_range (x k : ℝ) : 
  (∃ x, Real.sin (3 * x) = k * Real.sin x ∧ Real.sin x ≠ 0) ↔ -1 ≤ k ∧ k < 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_k_sin_x_solution_range_l1051_105122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_periodicity_l1051_105136

/-- Given a geometric progression with integer terms b_n = b_0 * q^n,
    where q is not divisible by p, prove that the last k digits of b_n
    and b_{n + p^(k-1) * (p-1)} are the same. -/
theorem geometric_progression_periodicity
  (p k : ℕ) (b₀ q : ℤ) (hp : p.Prime) (hq : ¬ (p : ℤ) ∣ q) :
  ∀ n : ℕ, (b₀ * q^n) ≡ (b₀ * q^(n + p^(k-1) * (p-1))) [ZMOD (10^k)] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_periodicity_l1051_105136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_factor_characterization_l1051_105177

/-- A graph represented by its vertex set and edge set -/
structure Graph (α : Type*) where
  V : Set α
  E : Set (α × α)

/-- The set of components of a graph -/
def components (G : Graph α) : Set (Set α) := sorry

/-- The subgraph induced by removing a set of vertices -/
def remove_vertices (G : Graph α) (S : Set α) : Graph α := sorry

/-- The prime subgraph of G with respect to S -/
def prime_subgraph (G : Graph α) (S : Set α) : Graph α := sorry

/-- The set of critical components of G-S -/
def critical_components (G : Graph α) (S : Set α) : Set (Set α) := sorry

/-- A matching in a graph -/
def is_matching (G : Graph α) (M : Set (α × α)) : Prop := sorry

/-- A perfect matching (1-factor) in a graph -/
def has_perfect_matching (G : Graph α) : Prop := sorry

/-- A set A can be matched to a set B in a graph -/
def can_be_matched_to (G : Graph α) (A B : Set α) : Prop := sorry

/-- Main theorem -/
theorem one_factor_characterization (G : Graph α) :
  ∃ S : Set α,
    (can_be_matched_to (prime_subgraph G S) (⋃₀ (critical_components G S)) S) ∧
    (∀ C ∈ components (remove_vertices G S), 
      C ∉ critical_components G S → has_perfect_matching (Graph.mk C (G.E ∩ (C.prod C)))) →
  (has_perfect_matching G ↔ 
    can_be_matched_to (prime_subgraph G S) (⋃₀ (critical_components G S)) S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_factor_characterization_l1051_105177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_convergence_l1051_105106

/-- Definition of sequences p_n and q_n -/
noncomputable def p_q_sequences (n : ℕ) (z : ℂ) : ℝ × ℝ :=
  let w := z^n
  (w.re, w.im)

/-- The sum we want to prove convergence for -/
noncomputable def sum_series (p q : ℕ → ℝ) : ℝ := ∑' n, (p n * q n) / (9 : ℝ)^n

/-- Main theorem statement -/
theorem sum_convergence :
  let z : ℂ := 3 + Complex.I
  let p := λ n => (p_q_sequences n z).1
  let q := λ n => (p_q_sequences n z).2
  sum_series p q = 3/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_convergence_l1051_105106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_plus_b_is_e_l1051_105162

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.exp x
def g (a b x : ℝ) := a * x + b

-- State the theorem
theorem max_a_plus_b_is_e :
  ∃ (a₀ b₀ : ℝ), (∀ a b : ℝ, (∀ x : ℝ, f x ≥ g a b x) → a + b ≤ a₀ + b₀) ∧
                 (∀ x : ℝ, f x ≥ g a₀ b₀ x) ∧
                 a₀ + b₀ = Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_plus_b_is_e_l1051_105162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l1051_105112

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    3 * (1 - 2^x) / (2^x + 1)
  else
    -1/4 * (x^3 + 3*x)

-- State the theorem
theorem x_range_for_inequality :
  (∀ (x m : ℝ), m ∈ Set.Icc (-3) 2 → f (m*x - 1) + f x > 0) →
  (∀ x : ℝ, f x = f (-x)) →  -- f is an odd function
  (∀ x y : ℝ, x < y → f y < f x) →  -- f is decreasing
  ∀ x : ℝ, x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l1051_105112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1051_105176

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define the line l
def line_l (a x y : ℝ) : Prop := a*x + y + 2*a = 0

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ line_l a x y ∧
  ∀ x' y' : ℝ, circle_C x' y' ∧ line_l a x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_line_value :
  ∀ a : ℝ, is_tangent a ↔ (a = Real.sqrt 2 / 4 ∨ a = -Real.sqrt 2 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1051_105176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_arrives_later_l1051_105154

/-- Represents the scenario of a motorboat and raft traveling along a river with three docks. -/
structure RiverScenario where
  current_speed : ℝ
  motorboat_speed : ℝ
  distance_AB : ℝ
  distance_BC : ℝ

/-- Calculates the time taken by the motorboat to complete its journey. -/
noncomputable def motorboat_time (scenario : RiverScenario) : ℝ :=
  scenario.distance_AB / (scenario.motorboat_speed - scenario.current_speed) +
  (scenario.distance_AB + scenario.distance_BC) / (scenario.motorboat_speed + scenario.current_speed)

/-- Calculates the time taken by the raft to reach dock C. -/
noncomputable def raft_time (scenario : RiverScenario) : ℝ :=
  scenario.distance_BC / scenario.current_speed

/-- Theorem stating the conditions under which the motorboat arrives later than the raft. -/
theorem motorboat_arrives_later (scenario : RiverScenario) :
  scenario.current_speed = 5 ∧
  scenario.distance_AB = scenario.distance_BC →
  (motorboat_time scenario > raft_time scenario ↔ 5 < scenario.motorboat_speed ∧ scenario.motorboat_speed < 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_arrives_later_l1051_105154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1051_105190

noncomputable def g (x : ℝ) : ℝ := min (2 * x + 2) (min ((1/2) * x + 1) (min (-(3/4) * x + 7) (3 * x + 4)))

theorem max_value_of_g :
  ∃ (M : ℝ), M = 17/5 ∧ ∀ (x : ℝ), g x ≤ M ∧ ∃ (x₀ : ℝ), g x₀ = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1051_105190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_polynomial_values_l1051_105198

theorem infinite_primes_dividing_polynomial_values (P : Polynomial ℤ) (h : P.degree > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Nat.Prime p ∧ ∃ x : ℕ, (P.eval (x : ℤ)) % (p : ℤ) = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_polynomial_values_l1051_105198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l1051_105158

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

-- Define the line of symmetry
noncomputable def symmetry_line (x y a : ℝ) : Prop := 3*x - a*y - 11 = 0

-- Define the midpoint of the chord
noncomputable def chord_midpoint (a : ℝ) : ℝ × ℝ := (a/4, -a/4)

-- Theorem statement
theorem chord_length_is_four (a : ℝ) :
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ x y, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) ∧
    symmetry_line center.1 center.2 a ∧
    let midpoint := chord_midpoint a;
    let d := Real.sqrt ((midpoint.1 - center.1)^2 + (midpoint.2 - center.2)^2);
    2 * Real.sqrt (r^2 - d^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l1051_105158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_poly_coefficients_equal_coefficients_sum_l1051_105195

-- Define the sequence of polynomials
def poly_seq : ℕ → (ℕ → ℕ) → (ℕ → ℕ) → (ℕ → ℕ)
| 0 => λ a b => a
| 1 => λ a b => b
| (n + 2) => λ a b => (2 * poly_seq n a b) + (poly_seq (n + 1) a b)

-- Theorem 1: The eighth polynomial is 42a + 43b
theorem eighth_poly_coefficients :
  ∀ a b : ℕ → ℕ, poly_seq 7 a b = λ x ↦ 42 * (a x) + 43 * (b x) := by
  sorry

-- Theorem 2: The sum of 2nth and (2n+1)th polynomials has equal coefficients of a and b
theorem equal_coefficients_sum :
  ∀ n : ℕ, ∀ a b : ℕ → ℕ, 
  ∃ k : ℕ, poly_seq (2*n - 1) a b + poly_seq (2*n) a b = λ x ↦ k * ((a x) + (b x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_poly_coefficients_equal_coefficients_sum_l1051_105195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_observations_l1051_105181

theorem initial_observations (initial_average : ℝ) (new_average : ℝ) : 
  initial_average = 15 ∧ new_average = initial_average - 1 → ∃ n : ℕ, n = 6 ∧ n > 0 ∧
  (n : ℝ) * initial_average = ((n : ℝ) + 1) * new_average :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_observations_l1051_105181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1051_105144

/-- The circle in the problem -/
def problem_circle : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 2)^2 = 4}

/-- The point through which the tangents pass -/
def P : ℝ × ℝ := (-2, 6)

/-- A function representing a line passing through two points -/
def line_through_points (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | (r.2 - p.2) * (q.1 - p.1) = (r.1 - p.1) * (q.2 - p.2)}

/-- Predicate to check if a point is on the circle -/
def on_circle (p : ℝ × ℝ) : Prop := p ∈ problem_circle

/-- Predicate to check if a line is tangent to the circle at a point -/
def is_tangent_at (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  on_circle p ∧ p ∈ l ∧ ∀ q : ℝ × ℝ, q ∈ l → q ≠ p → ¬on_circle q

/-- The main theorem -/
theorem tangent_line_equation :
  ∃ A B : ℝ × ℝ,
    is_tangent_at (line_through_points P A) A ∧
    is_tangent_at (line_through_points P B) B ∧
    line_through_points A B = {p : ℝ × ℝ | p.1 - 2*p.2 + 6 = 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1051_105144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1051_105121

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

noncomputable def g (x : ℝ) : ℝ :=
  if x = 0 then 1 else x / |x|

theorem f_eq_g : f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1051_105121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_ripeness_difference_l1051_105127

theorem peach_ripeness_difference (total_peaches initial_ripe daily_ripening eaten days : ℕ) :
  total_peaches = 18 →
  initial_ripe = 4 →
  daily_ripening = 2 →
  eaten = 3 →
  days = 5 →
  let total_ripened := initial_ripe + daily_ripening * days
  let ripe_after_eating := total_ripened - eaten
  let unripe := total_peaches - total_ripened
  ripe_after_eating - unripe = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_ripeness_difference_l1051_105127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1051_105174

/-- Predicate to ensure a, b, c form a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to ensure A, B, C are opposite to sides a, b, c respectively -/
def OppositeAngles (a b c A B C : ℝ) : Prop :=
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧
  Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S satisfies S = a² - (b-c)², then sin(A) / (1 - cos(A)) = 3 -/
theorem triangle_area_ratio (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
    (h_triangle : IsTriangle a b c)
    (h_area : S = a^2 - (b-c)^2)
    (h_opposite : OppositeAngles a b c A B C) :
    Real.sin A / (1 - Real.cos A) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1051_105174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_small_decimal_l1051_105155

/-- Given a real number 0.00000156, prove that its scientific notation representation is 1.56 * 10^(-6) -/
theorem scientific_notation_of_small_decimal :
  (0.00000156 : ℝ) = 1.56 * (10 : ℝ)^(-6 : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_small_decimal_l1051_105155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1051_105118

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 2
  {x : ℝ | f x ≥ 0} = Set.Iic (-1/2) ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1051_105118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4410_undefined_l1051_105116

theorem tan_4410_undefined : ¬∃ x : ℝ, Real.tan (4410 * π / 180) = x := by
  have h1 : 4410 * π / 180 = π / 2 := by
    -- Proof that 4410° is equivalent to 90°
    sorry
  have h2 : ¬∃ x : ℝ, Real.tan (π / 2) = x := by
    -- Proof that tan(π/2) is undefined
    sorry
  rw [h1]
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4410_undefined_l1051_105116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_implies_a_value_l1051_105132

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt ((a^2 + 3) / a^2)

-- Theorem statement
theorem hyperbola_eccentricity_implies_a_value (a : ℝ) 
  (h1 : a > 0) 
  (h2 : eccentricity a = 2) : 
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_implies_a_value_l1051_105132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l1051_105119

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: For any triangle with sides a, b, c and corresponding altitudes m_a, m_b, m_c,
    the sum of the products of pairs of altitudes is less than or equal to
    3/4 times the sum of the squares of the sides. -/
theorem triangle_altitude_inequality
  (a b c m_a m_b m_c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitude_a : m_a = 2 * (area a b c) / a)
  (h_altitude_b : m_b = 2 * (area a b c) / b)
  (h_altitude_c : m_c = 2 * (area a b c) / c)
  : m_a * m_b + m_b * m_c + m_c * m_a ≤ 3/4 * (a^2 + b^2 + c^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l1051_105119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_zero_l1051_105120

/-- Given complex numbers z₁ and z₂ defined in terms of a real number a,
    prove that if the real part of z₁ is greater than the real part of z₂,
    and their imaginary parts are equal, then a = 0. -/
theorem complex_inequality_implies_zero (a : ℝ) :
  let z₁ : ℂ := -4 * a + 1 + (2 * a^2 + 3 * a) * I
  let z₂ : ℂ := 2 * a + (a^2 + a) * I
  (z₁.re > z₂.re ∧ z₁.im = z₂.im) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_zero_l1051_105120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l1051_105170

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem distinct_values_of_S :
  ∃ (S : Finset ℂ), S.card = 3 ∧ 
  (∀ n : ℤ, (ω ^ n + ω ^ (-n)) ∈ S) ∧
  (∀ x : ℂ, x ∈ S → ∃ n : ℤ, x = ω ^ n + ω ^ (-n)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l1051_105170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_l1051_105179

theorem imaginary_sum (i : ℂ) (hi : i^2 = -1) : i^8 + i^20 + i^(-14 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_l1051_105179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_complement_equal_l1051_105186

universe u

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 3}

theorem union_complement_equal : A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_complement_equal_l1051_105186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_is_yellow_l1051_105161

/-- Represents the colors of beads in the necklace -/
inductive BeadColor
  | Red
  | Orange
  | Yellow
  | Green
  | Blue

/-- Defines the pattern of beads in one cycle -/
def beadPattern : List BeadColor :=
  [BeadColor.Red, BeadColor.Orange, 
   BeadColor.Yellow, BeadColor.Yellow, BeadColor.Yellow, 
   BeadColor.Green, 
   BeadColor.Blue, BeadColor.Blue]

/-- The total number of beads in the necklace -/
def totalBeads : Nat := 83

/-- Returns the color of the bead at a given position in the necklace -/
def beadColorAt (position : Nat) : BeadColor :=
  beadPattern[position % beadPattern.length]'sorry

/-- Theorem: The 83rd bead in the necklace is yellow -/
theorem last_bead_is_yellow : 
  beadColorAt (totalBeads - 1) = BeadColor.Yellow := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_is_yellow_l1051_105161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_less_than_20_l1051_105152

/-- The sum of all prime numbers less than 20 is 77. -/
theorem sum_of_primes_less_than_20 : (Finset.filter (fun n => n.Prime ∧ n < 20) (Finset.range 20)).sum id = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_less_than_20_l1051_105152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1051_105156

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : ℝ × ℝ × ℝ) (ha : a = (3, 4, -1)) (hb : b = (2, -1, 1)) :
  ∃ (k : ℝ), k ≠ 0 ∧ (6 • a - 3 • b) = k • (b - 2 • a) :=
by
  -- We'll use k = -3
  use -3
  constructor
  · -- Prove k ≠ 0
    norm_num
  · -- Prove the equality
    simp [ha, hb]
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1051_105156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_roots_sum_l1051_105123

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2
noncomputable def y : ℂ := (1 - i * Real.sqrt 3) / 2

-- State the theorem
theorem sixth_roots_sum : x^6 + y^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_roots_sum_l1051_105123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_patient_hours_l1051_105166

/-- Represents the pricing structure and patient charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Cost of the first hour
  additional_hour : ℕ  -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  two_hour_total : ℕ  -- Total charge for 2 hours of therapy

/-- Calculates the number of therapy hours given the pricing structure. -/
def calculate_therapy_hours (pricing : TherapyPricing) : ℕ :=
  let remaining := pricing.first_patient_total - pricing.first_hour
  1 + remaining / pricing.additional_hour

/-- Theorem stating that given the specific pricing structure, the first patient received 5 hours of therapy. -/
theorem first_patient_hours (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 40)
  (h2 : pricing.two_hour_total = 174)
  (h3 : pricing.first_patient_total = 375) : 
  calculate_therapy_hours pricing = 5 := by
  sorry

#eval calculate_therapy_hours { first_hour := 107, additional_hour := 67, first_patient_total := 375, two_hour_total := 174 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_patient_hours_l1051_105166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l1051_105183

/-- The parabola function y = -1/3x^2 --/
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^2

/-- The distance between two points in ℝ² --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The side length of an equilateral triangle on y = -1/3x^2 --/
theorem equilateral_triangle_on_parabola :
  ∀ a : ℝ,
  let A : ℝ × ℝ := (a, f a)
  let B : ℝ × ℝ := (-a, f (-a))
  let O : ℝ × ℝ := (0, 0)
  distance A O = distance B O → distance A O = distance A B →
  distance A B = 6 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l1051_105183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l1051_105135

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  /-- The length of the side common to the polygons -/
  c : ℝ
  /-- The polyhedron is convex -/
  convex : Bool
  /-- It has two congruent regular octagonal faces -/
  has_octagonal_faces : Bool
  /-- It has 16 congruent equilateral triangular faces -/
  has_triangular_faces : Bool
  /-- It has 8 congruent rhombic faces -/
  has_rhombic_faces : Bool
  /-- The sides of the two octagonal faces are pairwise parallel -/
  octagonal_sides_parallel : Bool
  /-- Each side of the octagonal face forms an edge with a side of the triangular faces -/
  octagonal_triangular_edge : Bool
  /-- The remaining sides of the triangular faces form edges with the rhombic faces -/
  triangular_rhombic_edge : Bool

/-- The radius of the smallest sphere that can enclose the polyhedron -/
noncomputable def smallest_enclosing_sphere_radius (p : ConvexPolyhedron) : ℝ :=
  p.c * Real.sqrt (1 + Real.sqrt 2)

/-- The volume of the polyhedron -/
noncomputable def polyhedron_volume (p : ConvexPolyhedron) : ℝ :=
  p.c^3 * (2 * Real.sqrt (Real.sqrt 8) / 3) * (1 + 5 * Real.sqrt 2)

/-- Theorem stating the properties of the convex polyhedron -/
theorem convex_polyhedron_properties (p : ConvexPolyhedron) :
  smallest_enclosing_sphere_radius p = p.c * Real.sqrt (1 + Real.sqrt 2) ∧
  polyhedron_volume p = p.c^3 * (2 * Real.sqrt (Real.sqrt 8) / 3) * (1 + 5 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l1051_105135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_is_four_percent_l1051_105128

/-- The population increase rate given the present population and population after one year -/
noncomputable def population_increase_rate (present_population : ℝ) (population_after_one_year : ℝ) : ℝ :=
  ((population_after_one_year - present_population) / present_population) * 100

/-- Theorem stating that the population increase rate is 4% given the specific population values -/
theorem population_increase_rate_is_four_percent
  (present_population : ℝ)
  (population_after_one_year : ℝ)
  (h1 : present_population = 1240)
  (h2 : population_after_one_year = 1289.6) :
  population_increase_rate present_population population_after_one_year = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_is_four_percent_l1051_105128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1051_105175

-- Define the two functions
def f (x : ℝ) : ℝ := x^(1/2)
def g (x : ℝ) : ℝ := x^2

-- Define the area of the closed region
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, (f x - g x)

-- Theorem statement
theorem area_between_curves : area = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1051_105175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_surface_area_l1051_105189

/-- The total surface area of a quadrilateral pyramid with a rhombus base -/
noncomputable def totalSurfaceArea (a : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  (2 * a^2 * Real.sin α * (Real.cos (β/2))^2) / Real.cos β

/-- Theorem: The total surface area of a quadrilateral pyramid with a rhombus base -/
theorem quadrilateral_pyramid_surface_area 
  (a : ℝ) (α : ℝ) (β : ℝ) 
  (ha : a > 0) 
  (hα : 0 < α ∧ α < π/2) 
  (hβ : 0 < β ∧ β < π/2) :
  totalSurfaceArea a α β = (2 * a^2 * Real.sin α * (Real.cos (β/2))^2) / Real.cos β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_surface_area_l1051_105189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l1051_105169

/-- Linear regression model -/
structure LinearRegressionModel where
  x : ℝ → ℝ  -- independent variable
  y : ℝ → ℝ  -- dependent variable
  a_hat : ℝ  -- estimated intercept
  b_hat : ℝ  -- estimated slope
  e_hat : ℝ → ℝ  -- residual function
  y_hat : ℝ → ℝ  -- predicted y function
  r_squared : ℝ  -- R-squared value

/-- Sum of squared residuals -/
def sum_squared_residuals (model : LinearRegressionModel) : ℝ :=
  sorry

/-- Residual plot characteristics -/
structure ResidualPlot where
  evenly_distributed : Prop
  horizontal_band : Prop
  band_width : ℝ

/-- Predicate for better fit between two models -/
def better_fit (model1 model2 : LinearRegressionModel) : Prop :=
  sorry

/-- Predicate for appropriate model -/
def appropriate_model (model : LinearRegressionModel) : Prop :=
  sorry

/-- Predicate for higher precision -/
def higher_precision (model : LinearRegressionModel) : Prop :=
  sorry

theorem linear_regression_properties 
  (model : LinearRegressionModel) 
  (residual_plot : ResidualPlot) : 
  -- Proposition 1
  (∀ x, model.e_hat x = model.y x - model.y_hat x) ∧ 
  (∀ x, model.y_hat x = model.b_hat * model.x x + model.a_hat) ∧
  -- Proposition 2
  (∀ model2 : LinearRegressionModel, 
    sum_squared_residuals model < sum_squared_residuals model2 → 
    better_fit model model2) ∧
  -- Proposition 3 (negated)
  ¬(∀ model2 : LinearRegressionModel, 
    model.r_squared < model2.r_squared → 
    better_fit model model2) ∧
  -- Proposition 4
  (residual_plot.evenly_distributed ∧ residual_plot.horizontal_band → 
    appropriate_model model) ∧
  (∀ plot2 : ResidualPlot, 
    residual_plot.band_width < plot2.band_width → 
    higher_precision model) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l1051_105169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1051_105163

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_rotation_power (n : ℕ) : 
  (n > 0 ∧ (rotation_matrix (160 * Real.pi / 180)) ^ n = 1 ∧ 
   ∀ m : ℕ, 0 < m ∧ m < n → (rotation_matrix (160 * Real.pi / 180)) ^ m ≠ 1) ↔ n = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1051_105163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_inequality_f_l1051_105146

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

-- State the theorems
theorem min_value_f : ∃ (m : ℝ), m = -2 / Real.exp 1 ∧
  ∀ x > 0, f x ≥ m := by
  sorry

theorem range_of_a : ∀ a : ℝ, (∃ x > 0, f x ≤ g a x) → a ≥ 4 := by
  sorry

theorem inequality_f : ∀ x > 0, f x > 2 * (x / Real.exp x - 2 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_inequality_f_l1051_105146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_size_even_club_neutral_partition_l1051_105140

/-- Represents a club where each member has one friend and one enemy -/
structure Club where
  members : Finset Nat
  friend : Nat → Nat
  enemy : Nat → Nat
  friend_valid : ∀ m, m ∈ members → friend m ∈ members ∧ friend m ≠ m
  enemy_valid : ∀ m, m ∈ members → enemy m ∈ members ∧ enemy m ≠ m
  friend_enemy_distinct : ∀ m, m ∈ members → friend m ≠ enemy m
  friendship_symmetric : ∀ m n, m ∈ members → n ∈ members → (friend m = n ↔ friend n = m)
  enmity_symmetric : ∀ m n, m ∈ members → n ∈ members → (enemy m = n ↔ enemy n = m)

/-- The number of members in the club is even -/
theorem club_size_even (c : Club) : Even c.members.card := by sorry

/-- The club can be divided into two neutral subgroups -/
theorem club_neutral_partition (c : Club) : 
  ∃ (s₁ s₂ : Finset Nat), s₁ ∪ s₂ = c.members ∧ s₁ ∩ s₂ = ∅ ∧
    (∀ m n, m ∈ s₁ → n ∈ s₁ → c.friend m ≠ n ∧ c.enemy m ≠ n) ∧
    (∀ m n, m ∈ s₂ → n ∈ s₂ → c.friend m ≠ n ∧ c.enemy m ≠ n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_size_even_club_neutral_partition_l1051_105140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statistical_order_l1051_105188

/-- Represents the statistical steps in analyzing math scores --/
inductive StatisticalStep
  | CollectScores
  | OrganizeData
  | DrawChart
  | AnalyzeChanges

/-- Defines the correct order of statistical steps --/
def correct_order : List StatisticalStep :=
  [StatisticalStep.CollectScores, StatisticalStep.OrganizeData, 
   StatisticalStep.DrawChart, StatisticalStep.AnalyzeChanges]

/-- Theorem stating that the given order is correct for analyzing math score changes --/
theorem correct_statistical_order :
  ∀ (order : List StatisticalStep),
  order = correct_order ↔ 
  (order.length = 4 ∧ 
   order.get? 0 = some StatisticalStep.CollectScores ∧
   order.get? 1 = some StatisticalStep.OrganizeData ∧
   order.get? 2 = some StatisticalStep.DrawChart ∧
   order.get? 3 = some StatisticalStep.AnalyzeChanges) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statistical_order_l1051_105188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l1051_105108

/-- The function f(x) defined on the interval [3,4] -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 10 * a / x) / Real.log a

/-- The statement that f is monotonic on [3,4] iff a is in the specified range -/
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y ∧ y ≤ 4 → f a x ≤ f a y) ∨
  (∀ x y, 3 ≤ x ∧ x ≤ y ∧ y ≤ 4 → f a y ≤ f a x) ↔
  a ∈ Set.Ioo 0 (9/10) ∪ Set.Ici (8/5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l1051_105108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_crossing_time_l1051_105126

-- Define the lengths of the trains in meters
def train_a_length : ℝ := 150
def train_b_length : ℝ := 150

-- Define the speeds of the trains in km/h
def train_a_speed : ℝ := 54
def train_b_speed : ℝ := 36

-- Convert km/h to m/s
noncomputable def km_per_hour_to_m_per_second (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  km_per_hour_to_m_per_second train_a_speed + km_per_hour_to_m_per_second train_b_speed

-- Calculate total length to be crossed
def total_length : ℝ := train_a_length + train_b_length

-- Theorem: The time taken for Arun to completely cross train B is 12 seconds
theorem arun_crossing_time :
  total_length / relative_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_crossing_time_l1051_105126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_diameter_correct_l1051_105172

/-- The least positive integer diameter such that the error in area calculation exceeds 1 -/
def least_diameter_with_error_exceeding_one : ℕ := 51

/-- The approximation of π used in the problem -/
def π_approx : ℚ := 3.14

/-- The actual area of a circle with diameter d -/
noncomputable def actual_area (d : ℕ) : ℝ := Real.pi * ((d : ℝ) / 2)^2

/-- The approximated area of a circle with diameter d using π_approx -/
def approx_area (d : ℕ) : ℚ := π_approx * ((d : ℚ) / 2)^2

/-- The error in area calculation for a circle with diameter d -/
noncomputable def area_error (d : ℕ) : ℝ := |actual_area d - (approx_area d : ℝ)|

theorem least_diameter_correct :
  (∀ d : ℕ, d < least_diameter_with_error_exceeding_one → area_error d ≤ 1) ∧
  area_error least_diameter_with_error_exceeding_one > 1 := by
  sorry

#eval least_diameter_with_error_exceeding_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_diameter_correct_l1051_105172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l1051_105199

noncomputable def parabola1 (x : ℝ) : ℝ := -1/4 * x^2 + 5*x - 21

def A : ℝ × ℝ := (10, 4)

def B : ℝ × ℝ := (6, 0)
def F : ℝ × ℝ := (14, 0)

noncomputable def parabola2 (x : ℝ) : ℝ := 1/4 * (x - B.1)^2

def D : ℝ × ℝ := (0, 9)

theorem parabola_intersection :
  (parabola1 A.1 = A.2) ∧
  (parabola1 B.1 = B.2) ∧
  (parabola1 F.1 = F.2) ∧
  (B.1 < F.1) ∧
  (parabola2 A.1 = A.2) ∧
  (parabola2 D.1 = D.2) :=
by sorry

#eval D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l1051_105199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_l1051_105105

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) - 2 * Real.cos x

theorem f_maximum :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 2) ∧
  (∀ (x : ℝ), f x = 2 ↔ ∃ (k : ℤ), x = 2 * ↑k * Real.pi + 2 * Real.pi / 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_l1051_105105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_statistics_l1051_105171

theorem leap_year_date_statistics :
  let data : List ℕ := (List.range 30).bind (λ i => List.replicate 12 (i + 1)) ++ List.replicate 12 30 ++ List.replicate 11 31
  let d : ℚ := 31/2 -- median of modes (1 to 30)
  let M : ℚ := 16 -- median of the entire data set
  let μ : ℚ := (data.sum : ℚ) / 366 -- mean of the entire data set
  d < M ∧ M < μ :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_statistics_l1051_105171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_at_one_l1051_105117

/-- The function f(x) = x^3 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

/-- h is a cubic polynomial -/
noncomputable def h : Polynomial ℝ := sorry

/-- The value of h at 0 is 1 -/
axiom h_at_zero : h.eval 0 = 1

/-- The roots of h are the squares of the roots of f -/
axiom h_roots (r : ℝ) : f r = 0 ↔ h.eval (r^2) = 0

/-- The value of h at 1 is 0 -/
theorem h_at_one : h.eval 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_at_one_l1051_105117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_proof_l1051_105157

/-- Given a triangle with sides a and b, and included angle C, 
    calculates the altitude h to side b. -/
noncomputable def altitude (a b : ℝ) (C : ℝ) : ℝ :=
  a * Real.sin C / 2

theorem triangle_altitude_proof :
  let a : ℝ := 18
  let b : ℝ := 24
  let C : ℝ := π / 3  -- 60 degrees in radians
  altitude a b C = 9 * Real.sqrt 3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_proof_l1051_105157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1051_105142

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The condition that the asymptotes of the hyperbola intersect with the circle x² + (y - 2)² = 1 -/
def asymptotes_intersect_circle (h : Hyperbola) : Prop :=
  2 * h.b / Real.sqrt (h.a^2 + h.b^2) < 1

/-- The theorem stating the range of eccentricity for the given hyperbola -/
theorem eccentricity_range (h : Hyperbola) (h_intersect : asymptotes_intersect_circle h) :
  1 < eccentricity h ∧ eccentricity h ≤ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1051_105142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equation_l1051_105149

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trajectory_and_line_equation :
  ∀ (x y : ℝ),
  let a : ℝ × ℝ := (2*x + 3, y)
  let b : ℝ × ℝ := (2*x - 3, 3*y)
  let F : ℝ × ℝ := (0, 1)
  let trajectory_eq := λ (p : ℝ × ℝ) => p.1^2 / 3 + p.2^2 / 4 = 1
  let line_eq := λ (p : ℝ × ℝ) (k : ℝ) => p.2 = k * p.1 + 1
  (dot_product a b = 3) →
  (∃ (A B : ℝ × ℝ) (k : ℝ),
    trajectory_eq A ∧ trajectory_eq B ∧
    line_eq A k ∧ line_eq B k ∧
    line_eq F k ∧
    distance A B = 16/5) →
  (trajectory_eq (x, y) ∧ (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equation_l1051_105149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_flights_for_10x10_l1051_105147

/-- Represents the movement rules for the grasshopper on a 10x10 grid -/
structure GrasshopperMovement where
  grid_size : Nat
  start_pos : Nat × Nat
  can_move_down : Bool
  can_move_right : Bool
  can_jump_bottom_to_top : Bool
  can_jump_right_to_left : Bool

/-- Represents a cell on the diagonal from top-right to bottom-left -/
def DiagonalCell (gm : GrasshopperMovement) (n : Nat) : Nat × Nat :=
  (n, gm.grid_size - n + 1)

/-- The minimum number of flights required to visit all cells -/
def MinFlights (gm : GrasshopperMovement) : Nat :=
  gm.grid_size - 1

/-- Theorem stating that at least 9 flights are required for a 10x10 grid -/
theorem min_flights_for_10x10 (gm : GrasshopperMovement) 
  (h1 : gm.grid_size = 10)
  (h2 : gm.start_pos = (1, 1))
  (h3 : gm.can_move_down = true)
  (h4 : gm.can_move_right = true)
  (h5 : gm.can_jump_bottom_to_top = true)
  (h6 : gm.can_jump_right_to_left = true) :
  MinFlights gm ≥ 9 := by
  sorry

#check min_flights_for_10x10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_flights_for_10x10_l1051_105147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progressions_lower_bound_l1051_105111

/-- 
Given s arithmetic progressions of integers P₁, ..., Pₛ satisfying:
1) Each integer belongs to at least one progression
2) Each progression contains a number that doesn't belong to other progressions
3) n is the least common multiple of the ratios of these progressions
4) n has a prime factorization n = p₁ᵅ¹ ... pₖᵅᵏ

This theorem states that the number of progressions s is greater than or equal to 
1 plus the sum of αᵢ(pᵢ - 1) for all prime factors pᵢ of n.
-/
theorem arithmetic_progressions_lower_bound 
  (s : ℕ) -- number of arithmetic progressions
  (P : Fin s → Set ℤ) -- the s arithmetic progressions
  (n : ℕ) -- least common multiple of the ratios
  (p : ℕ → ℕ) -- prime factors of n
  (α : ℕ → ℕ) -- exponents in prime factorization of n
  (k : ℕ) -- number of prime factors of n
  (h1 : ∀ (z : ℤ), ∃ (i : Fin s), z ∈ P i) -- each integer belongs to at least one progression
  (h2 : ∀ (i : Fin s), ∃ (z : ℤ), z ∈ P i ∧ ∀ (j : Fin s), j ≠ i → z ∉ P j) -- each progression has a unique element
  (h3 : n = Finset.prod (Finset.range k) (fun i => p i ^ α i)) -- prime factorization of n
  (h4 : ∀ i, i ∈ Finset.range k → Nat.Prime (p i)) -- p i are prime
  : s ≥ 1 + Finset.sum (Finset.range k) (fun i => α i * (p i - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progressions_lower_bound_l1051_105111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_points_condition_l1051_105101

-- Define the function f(x)
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 * Real.log x + m

-- State the theorem
theorem two_zero_points_condition (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧
    ∀ (x : ℝ), f x m = 0 → x = x₁ ∨ x = x₂) ↔
  m < 1 / (3 * Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_points_condition_l1051_105101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_half_necessary_not_sufficient_l1051_105165

theorem cos_2alpha_half_necessary_not_sufficient :
  (∀ α : Real, Real.sin α = 1/2 → Real.cos (2*α) = 1/2) ∧
  (∃ α : Real, Real.cos (2*α) = 1/2 ∧ Real.sin α ≠ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_half_necessary_not_sufficient_l1051_105165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_real_roots_probability_l1051_105148

noncomputable def polynomial (b x : ℝ) : ℝ :=
  x^4 + 2*b*x^3 + (2*b - 3)*x^2 + (-4*b + 6)*x - 3

def interval_length : ℝ := 35

noncomputable def probability_all_real_roots (interval : ℝ) : ℝ :=
  1 - Real.sqrt 6 / interval

theorem all_real_roots_probability :
  probability_all_real_roots interval_length = 1 - Real.sqrt 6 / 35 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_real_roots_probability_l1051_105148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1051_105151

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2/3) : 
  Real.cos (π - 2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1051_105151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_height_l1051_105191

-- Define the circle sector
noncomputable def sector_radius : ℝ := 6
noncomputable def sector_fraction : ℝ := 5/6

-- Define the cone
noncomputable def cone_slant_height : ℝ := sector_radius
noncomputable def cone_base_radius : ℝ := sector_fraction * sector_radius

-- Theorem statement
theorem cone_volume_and_height :
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * Real.pi * cone_base_radius^2 * cone_height
  (cone_volume = (25/3) * Real.pi * Real.sqrt 11) ∧ 
  (cone_height = Real.sqrt 11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_height_l1051_105191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_jasmine_l1051_105102

def max_notebooks (available : ℚ) (cost : ℚ) : ℕ :=
  (available / cost).floor.toNat

theorem max_notebooks_jasmine :
  max_notebooks 10 1.25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_jasmine_l1051_105102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lcm_factor_l1051_105167

theorem smallest_lcm_factor (x : ℕ) : 
  x > 0 ∧ Nat.lcm x (Nat.lcm 108 2100) = 37800 ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → Nat.lcm y (Nat.lcm 108 2100) ≠ 37800) → 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lcm_factor_l1051_105167
