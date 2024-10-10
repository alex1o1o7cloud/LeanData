import Mathlib

namespace min_a_2005_l1126_112600

theorem min_a_2005 (a : Fin 2005 → ℕ+) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_product : ∀ i j k, i ≠ j → i < 2005 → j < 2005 → k < 2005 → a i * a j ≠ a k) :
  a 2004 ≥ 2048 := by
  sorry

end min_a_2005_l1126_112600


namespace distinct_collections_biology_l1126_112670

-- Define the set of letters in BIOLOGY
def biology : Finset Char := {'B', 'I', 'O', 'L', 'G', 'Y'}

-- Define the set of vowels in BIOLOGY
def vowels : Finset Char := {'I', 'O'}

-- Define the set of consonants in BIOLOGY
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

-- Define the number of vowels to be selected
def num_vowels : ℕ := 2

-- Define the number of consonants to be selected
def num_consonants : ℕ := 4

-- Define a function to count distinct collections
def count_distinct_collections : ℕ := sorry

-- Theorem statement
theorem distinct_collections_biology :
  count_distinct_collections = 2 :=
sorry

end distinct_collections_biology_l1126_112670


namespace max_pairs_sum_l1126_112601

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1504 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ) (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      m = pairs'.card →
      m ≤ k) :=
by sorry

end max_pairs_sum_l1126_112601


namespace cattle_transport_time_l1126_112676

/-- Calculates the total driving time to transport cattle to higher ground -/
def total_driving_time (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (speed : ℕ) : ℕ :=
  let num_trips := total_cattle / truck_capacity
  let round_trip_time := 2 * (distance / speed)
  num_trips * round_trip_time

/-- Theorem stating that under given conditions, the total driving time is 40 hours -/
theorem cattle_transport_time : total_driving_time 400 60 20 60 = 40 := by
  sorry

end cattle_transport_time_l1126_112676


namespace ratio_w_to_y_l1126_112607

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end ratio_w_to_y_l1126_112607


namespace min_value_expression_l1126_112681

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_expression_l1126_112681


namespace merchant_tea_cups_l1126_112619

theorem merchant_tea_cups (S O P : ℕ) 
  (h1 : S + O = 11) 
  (h2 : P + O = 15) 
  (h3 : S + P = 14) : 
  S + O + P = 20 := by
sorry

end merchant_tea_cups_l1126_112619


namespace smallest_d_for_factorization_l1126_112698

theorem smallest_d_for_factorization : 
  (∃ (p q : ℤ), x^2 + 107*x + 2050 = (x + p) * (x + q)) ∧ 
  (∀ (d : ℕ), d < 107 → ¬∃ (p q : ℤ), x^2 + d*x + 2050 = (x + p) * (x + q)) := by
  sorry

end smallest_d_for_factorization_l1126_112698


namespace length_PQ_is_two_l1126_112630

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a ray in polar form -/
structure PolarRay where
  θ : ℝ

theorem length_PQ_is_two 
  (l : ParametricLine)
  (C : PolarCircle)
  (OM : PolarRay)
  (h1 : l.x = fun t ↦ -1/2 * t)
  (h2 : l.y = fun t ↦ 3 * Real.sqrt 3 + (Real.sqrt 3 / 2) * t)
  (h3 : C.equation = fun ρ θ ↦ ρ = 2 * Real.cos θ)
  (h4 : OM.θ = π / 3)
  (P : PolarPoint)
  (Q : PolarPoint)
  (h5 : C.equation P.ρ P.θ)
  (h6 : P.θ = OM.θ)
  (h7 : Q.θ = OM.θ)
  (h8 : Real.sqrt 3 * Q.ρ * Real.cos Q.θ + Q.ρ * Real.sin Q.θ - 3 * Real.sqrt 3 = 0) :
  abs (P.ρ - Q.ρ) = 2 := by
sorry


end length_PQ_is_two_l1126_112630


namespace even_integers_between_fractions_l1126_112615

theorem even_integers_between_fractions :
  let lower_bound : ℚ := 23/5
  let upper_bound : ℚ := 47/3
  (Finset.filter (fun n => n % 2 = 0) (Finset.Icc ⌈lower_bound⌉ ⌊upper_bound⌋)).card = 5 :=
by sorry

end even_integers_between_fractions_l1126_112615


namespace ellipse_axis_endpoints_distance_l1126_112623

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis
    of the ellipse 4(x-2)^2 + 16(y-3)^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoints_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * (y - 3)^2 = 64 ↔ 
      (x - 2)^2 / 16 + (y - 3)^2 / 4 = 1) → 
    (C.1 - 2)^2 / 16 + (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + (D.2 - 3)^2 / 4 = 1 →
    (C.1 - 2)^2 / 16 = 1 ∨ (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 = 1 ∨ (D.2 - 3)^2 / 4 = 1 →
    ((C.1 - 2)^2 / 16 = 1 ∧ (D.2 - 3)^2 / 4 = 1) ∨
    ((C.2 - 3)^2 / 4 = 1 ∧ (D.1 - 2)^2 / 16 = 1) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoints_distance_l1126_112623


namespace belize_homes_without_features_l1126_112672

/-- The town of Belize with its home characteristics -/
structure BelizeTown where
  total_homes : ℕ
  white_homes : ℕ
  non_white_homes : ℕ
  non_white_with_fireplace : ℕ
  non_white_with_fireplace_and_basement : ℕ
  non_white_without_fireplace : ℕ
  non_white_without_fireplace_with_garden : ℕ

/-- Properties of the Belize town -/
def belize_properties (t : BelizeTown) : Prop :=
  t.total_homes = 400 ∧
  t.white_homes = t.total_homes / 4 ∧
  t.non_white_homes = t.total_homes - t.white_homes ∧
  t.non_white_with_fireplace = t.non_white_homes / 5 ∧
  t.non_white_with_fireplace_and_basement = t.non_white_with_fireplace / 3 ∧
  t.non_white_without_fireplace = t.non_white_homes - t.non_white_with_fireplace ∧
  t.non_white_without_fireplace_with_garden = t.non_white_without_fireplace / 2

/-- Theorem: The number of non-white homes without fireplace, basement, or garden is 120 -/
theorem belize_homes_without_features (t : BelizeTown) 
  (h : belize_properties t) : 
  t.non_white_without_fireplace - t.non_white_without_fireplace_with_garden = 120 := by
  sorry

end belize_homes_without_features_l1126_112672


namespace circumcenter_outside_l1126_112648

/-- An isosceles trapezoid with specific angle measurements -/
structure IsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The angle between the diagonals adjacent to the lateral side -/
  diagonal_angle : ℝ
  /-- Condition that the base angle is 50 degrees -/
  base_angle_eq : base_angle = 50
  /-- Condition that the diagonal angle is 40 degrees -/
  diagonal_angle_eq : diagonal_angle = 40

/-- The location of the circumcenter relative to the trapezoid -/
inductive CircumcenterLocation
  | Inside
  | Outside

/-- Theorem stating that the circumcenter is outside the trapezoid -/
theorem circumcenter_outside (t : IsoscelesTrapezoid) : 
  CircumcenterLocation.Outside = 
    CircumcenterLocation.Outside := by sorry

end circumcenter_outside_l1126_112648


namespace sum_of_2001_and_1015_l1126_112621

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end sum_of_2001_and_1015_l1126_112621


namespace additional_interest_rate_proof_l1126_112603

/-- Proves that given specific investment conditions, the additional interest rate must be 8% --/
theorem additional_interest_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (total_rate : ℝ) (additional_investment : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.04 →
  total_rate = 0.06 →
  additional_investment = 2400 →
  (initial_investment * initial_rate + additional_investment * 0.08) / 
    (initial_investment + additional_investment) = total_rate :=
by sorry

end additional_interest_rate_proof_l1126_112603


namespace function_and_tangent_line_l1126_112614

/-- Given a function f with the property that 
    f(x) = (1/4) * f'(1) * x^2 + 2 * f(1) * x - 4,
    prove that f(x) = 2x^2 + 4x - 4 and its tangent line
    at (0, f(0)) has the equation 4x - y - 4 = 0 -/
theorem function_and_tangent_line 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = (1/4) * (deriv f 1) * x^2 + 2 * (f 1) * x - 4) :
  (∀ x, f x = 2*x^2 + 4*x - 4) ∧ 
  (∃ a b c : ℝ, a = 4 ∧ b = -1 ∧ c = -4 ∧ 
    ∀ x y, y = (deriv f 0) * x + f 0 ↔ a*x + b*y + c = 0) :=
by sorry

end function_and_tangent_line_l1126_112614


namespace cubic_inequality_l1126_112641

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a+b) + b*c*(b+c) + c*a*(c+a) := by
  sorry

end cubic_inequality_l1126_112641


namespace jerry_remaining_money_l1126_112642

/-- Calculates the remaining money after expenses --/
def remaining_money (initial_amount video_games_cost snack_cost toy_original_cost toy_discount_percent : ℚ) : ℚ :=
  let toy_discount := toy_original_cost * (toy_discount_percent / 100)
  let toy_final_cost := toy_original_cost - toy_discount
  let total_spent := video_games_cost + snack_cost + toy_final_cost
  initial_amount - total_spent

/-- Theorem stating that Jerry's remaining money is $6 --/
theorem jerry_remaining_money :
  remaining_money 18 6 3 4 25 = 6 := by
  sorry

end jerry_remaining_money_l1126_112642


namespace min_product_xy_l1126_112683

theorem min_product_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end min_product_xy_l1126_112683


namespace exactlyOneOfThreeCount_l1126_112622

/-- The number of math majors taking exactly one of Galois Theory, Hyperbolic Geometry, or Topology -/
def exactlyOneOfThree (total : ℕ) (noElective : ℕ) (ant_gt : ℕ) (gt_hg : ℕ) (hg_cry : ℕ) (cry_top : ℕ) (top_ant : ℕ) (ant_or_cry : ℕ) : ℕ :=
  total - noElective - ant_gt - gt_hg - hg_cry - cry_top - top_ant - ant_or_cry

theorem exactlyOneOfThreeCount :
  exactlyOneOfThree 100 22 7 12 3 15 8 16 = 17 :=
sorry

end exactlyOneOfThreeCount_l1126_112622


namespace notebook_difference_l1126_112608

theorem notebook_difference (price : ℚ) (mika_count leo_count : ℕ) : 
  price > (1 / 10 : ℚ) →
  price * mika_count = (12 / 5 : ℚ) →
  price * leo_count = (16 / 5 : ℚ) →
  leo_count - mika_count = 4 := by
  sorry

#check notebook_difference

end notebook_difference_l1126_112608


namespace video_votes_l1126_112692

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) : 
  likes + dislikes = total_votes →
  likes = (3 * total_votes) / 4 →
  dislikes = total_votes / 4 →
  likes - dislikes = 50 →
  total_votes = 100 := by
sorry

end video_votes_l1126_112692


namespace tangent_line_at_M_l1126_112659

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define point M on the circle and the line y = -x
def point_M (x y : ℝ) : Prop := circle_C x y ∧ y = -x

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 2 - Real.sqrt 2

-- Theorem statement
theorem tangent_line_at_M :
  ∀ x y : ℝ, point_M x y → tangent_line x y :=
by sorry

end tangent_line_at_M_l1126_112659


namespace max_value_under_constraint_l1126_112605

theorem max_value_under_constraint (x y : ℝ) :
  (|x| + |y| ≤ 1) → (x + 2*y ≤ 2) :=
by sorry

end max_value_under_constraint_l1126_112605


namespace ellipse_problem_l1126_112666

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2 * (x + 1)

-- Define the perpendicular condition
def perpendicular_condition (M A : ℝ × ℝ) : Prop :=
  (A.2 - M.2) = -1/2 * (A.1 - M.1)

-- Theorem statement
theorem ellipse_problem :
  ∀ (P : ℝ × ℝ), distance_condition P →
  (∀ x y, trajectory_C x y ↔ (x, y) = P) ∧
  (∃ (M : ℝ × ℝ), trajectory_C M.1 M.2 ∧
    ∀ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A →
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) ≤ Real.sqrt 5) ∧
  (∃ (M : ℝ × ℝ), M = (1, 3/2) ∧ trajectory_C M.1 M.2 ∧
    ∃ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A ∧
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = Real.sqrt 5) :=
sorry

end ellipse_problem_l1126_112666


namespace triangle_cosine_theorem_l1126_112646

theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (3 * a * Real.cos A = c * Real.cos B + b * Real.cos C) →
  (Real.cos A = 1 / 3) ∧
  (a = 2 * Real.sqrt 3 ∧ Real.cos B + Real.cos C = 2 * Real.sqrt 3 / 3 → c = 3) :=
by sorry

end triangle_cosine_theorem_l1126_112646


namespace lower_rent_is_40_l1126_112640

/-- Represents the motel rental scenario -/
structure MotelRental where
  lower_rent : ℕ  -- The lower rent amount
  higher_rent : ℕ := 60  -- The higher rent amount, fixed at $60
  total_rent : ℕ := 1000  -- The total rent for the night
  reduction_percent : ℕ := 20  -- The reduction percentage if 10 rooms switch to lower rent

/-- Theorem stating that the lower rent amount is $40 -/
theorem lower_rent_is_40 (m : MotelRental) : m.lower_rent = 40 := by
  sorry

#check lower_rent_is_40

end lower_rent_is_40_l1126_112640


namespace angle_E_measure_l1126_112631

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ)

-- Define the conditions for the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  q.E = 3 * q.F ∧ q.E = 4 * q.G ∧ q.E = 6 * q.H ∧
  q.E + q.F + q.G + q.H = 360

-- Theorem statement
theorem angle_E_measure (q : Quadrilateral) 
  (h : is_special_quadrilateral q) : 
  205 < q.E ∧ q.E < 206 :=
sorry

end angle_E_measure_l1126_112631


namespace angle_beta_properties_l1126_112637

theorem angle_beta_properties (β : Real) 
  (h1 : π/2 < β ∧ β < π)  -- β is in the second quadrant
  (h2 : (2 * Real.tan β ^ 2) / (3 * Real.tan β + 2) = 1) :  -- β satisfies the given equation
  (Real.sin (β + 3 * π / 2) = 2 * Real.sqrt 5 / 5) ∧ 
  ((2 / 3) * Real.sin β ^ 2 + Real.cos β * Real.sin β = -1 / 15) := by
  sorry

end angle_beta_properties_l1126_112637


namespace infinitely_many_m_for_composite_sum_l1126_112690

theorem infinitely_many_m_for_composite_sum : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (m : ℕ+), m ∈ S → 
      ∀ (n : ℕ+), ∃ (a b : ℕ+), a * b = n^4 + m ∧ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end infinitely_many_m_for_composite_sum_l1126_112690


namespace arithmetic_sequence_property_l1126_112664

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 + a 5 = 8 →                                       -- given condition
  a 3 = 4 :=                                            -- conclusion to prove
by sorry

end arithmetic_sequence_property_l1126_112664


namespace polynomial_simplification_l1126_112661

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 - 5*x + 42) + (2*x - 7)*(x - 2)*(x + 3) = 
  3*x^3 - 4*x^2 - 62*x + 116 := by
  sorry

end polynomial_simplification_l1126_112661


namespace simplify_complex_fraction_l1126_112694

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 4) 
  (h2 : x ≠ 2) 
  (h3 : x ≠ 5) 
  (h4 : x ≠ 3) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3)^2 / ((x - 4) * (x - 2)) := by
  sorry

end simplify_complex_fraction_l1126_112694


namespace spratilish_word_count_mod_1000_l1126_112663

/-- Represents a Spratilish letter -/
inductive SpratilishLetter
| M
| P
| Z
| O

/-- Checks if a SpratilishLetter is a consonant -/
def isConsonant (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.M => true
  | SpratilishLetter.P => true
  | _ => false

/-- Checks if a SpratilishLetter is a vowel -/
def isVowel (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.Z => true
  | SpratilishLetter.O => true
  | _ => false

/-- Represents a Spratilish word as a list of SpratilishLetters -/
def SpratilishWord := List SpratilishLetter

/-- Checks if a SpratilishWord is valid (at least three consonants between any two vowels) -/
def isValidSpratilishWord (w : SpratilishWord) : Bool :=
  sorry

/-- Counts the number of valid 9-letter Spratilish words -/
def countValidSpratilishWords : Nat :=
  sorry

/-- The main theorem: The number of valid 9-letter Spratilish words is congruent to 704 modulo 1000 -/
theorem spratilish_word_count_mod_1000 :
  countValidSpratilishWords % 1000 = 704 := by sorry

end spratilish_word_count_mod_1000_l1126_112663


namespace hearts_ratio_equals_half_l1126_112679

-- Define the ♥ operation
def hearts (n m : ℕ) : ℕ := n^4 * m^3

-- Theorem statement
theorem hearts_ratio_equals_half : 
  (hearts 2 4) / (hearts 4 2) = 1 / 2 := by
  sorry

end hearts_ratio_equals_half_l1126_112679


namespace factor_theorem_quadratic_l1126_112658

theorem factor_theorem_quadratic (t : ℚ) : 
  (∀ x, (x - t) ∣ (4*x^2 + 9*x + 2)) ↔ (t = -1/4 ∨ t = -2) :=
by sorry

end factor_theorem_quadratic_l1126_112658


namespace quadratic_two_distinct_roots_l1126_112645

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*x + m - 1 = 0 ∧ 
    y^2 - 2*y + m - 1 = 0) →
  m < 2 := by
sorry

end quadratic_two_distinct_roots_l1126_112645


namespace fourth_player_win_probability_l1126_112610

/-- The probability of the fourth player winning in a coin-flipping game with four players -/
theorem fourth_player_win_probability :
  let p : ℕ → ℝ := λ n => (1 / 2) ^ (4 * n + 1)
  let total_prob := (∑' n, p n)
  total_prob = 1 / 30
  := by sorry

end fourth_player_win_probability_l1126_112610


namespace miss_graysons_class_fund_miss_graysons_class_fund_proof_l1126_112606

theorem miss_graysons_class_fund (initial_fund : ℕ) (student_contribution : ℕ) (num_students : ℕ) (trip_cost : ℕ) : ℕ :=
  let total_contribution := student_contribution * num_students
  let total_fund := initial_fund + total_contribution
  let total_trip_cost := trip_cost * num_students
  let remaining_fund := total_fund - total_trip_cost
  remaining_fund

theorem miss_graysons_class_fund_proof :
  miss_graysons_class_fund 50 5 20 7 = 10 := by
  sorry

end miss_graysons_class_fund_miss_graysons_class_fund_proof_l1126_112606


namespace river_crossing_problem_l1126_112682

/-- The minimum number of trips required to transport a group of people across a river -/
def min_trips (total_people : ℕ) (boat_capacity : ℕ) (boatman_required : Bool) : ℕ :=
  let effective_capacity := if boatman_required then boat_capacity - 1 else boat_capacity
  (total_people + effective_capacity - 1) / effective_capacity

theorem river_crossing_problem :
  min_trips 14 5 true = 4 := by
  sorry

end river_crossing_problem_l1126_112682


namespace centers_form_rectangle_l1126_112632

/-- Represents a quadrilateral with side lengths -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Checks if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the center of a rectangle given two adjacent corners -/
def rectangle_center (p1 p2 : Point) (width height : ℝ) : Point :=
  sorry

/-- Checks if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Main theorem: The centers of rectangles constructed on the sides of an inscribed quadrilateral form a rectangle -/
theorem centers_form_rectangle (q : Quadrilateral) (h : is_inscribed q) :
  let a := q.a
  let b := q.b
  let c := q.c
  let d := q.d
  let A := Point.mk 0 0  -- Arbitrary placement of A
  let B := Point.mk a 0  -- B is a units away from A on x-axis
  let C := sorry         -- C's position depends on the quadrilateral's shape
  let D := sorry         -- D's position depends on the quadrilateral's shape
  let P := rectangle_center A B a c
  let Q := rectangle_center B C b d
  let R := rectangle_center C D c a
  let S := rectangle_center D A d b
  is_rectangle P Q R S :=
sorry

end centers_form_rectangle_l1126_112632


namespace fifteenth_term_of_geometric_sequence_l1126_112651

/-- Given a geometric sequence with first term 32 and common ratio 1/4,
    prove that the 15th term is 1/8388608 -/
theorem fifteenth_term_of_geometric_sequence :
  let a₁ : ℚ := 32  -- First term
  let r : ℚ := 1/4  -- Common ratio
  let n : ℕ := 15   -- Term number we're looking for
  let aₙ : ℚ := a₁ * r^(n-1)  -- General term formula
  aₙ = 1/8388608 := by sorry

end fifteenth_term_of_geometric_sequence_l1126_112651


namespace inequality_satisfied_iff_m_in_range_l1126_112635

theorem inequality_satisfied_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ 
  m < -1/2 := by
sorry

end inequality_satisfied_iff_m_in_range_l1126_112635


namespace h_satisfies_condition_l1126_112674

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 4 * x - 5

def h (x : ℝ) : ℝ := 2 * x - 4

theorem h_satisfies_condition : ∀ x : ℝ, f (h x) = g x := by
  sorry

end h_satisfies_condition_l1126_112674


namespace eugene_pencils_l1126_112684

theorem eugene_pencils (x : ℕ) (h1 : x + 6 = 57) : x = 51 := by
  sorry

end eugene_pencils_l1126_112684


namespace fourth_root_of_256000000_l1126_112650

theorem fourth_root_of_256000000 : (400 : ℕ) ^ 4 = 256000000 := by
  sorry

end fourth_root_of_256000000_l1126_112650


namespace circle_x_axis_intersection_l1126_112678

/-- A circle with diameter endpoints at (0,0) and (10, -6) -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 3)^2 = 34}

/-- The x-axis -/
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- The point (0,0) -/
def Origin : ℝ × ℝ := (0, 0)

theorem circle_x_axis_intersection :
  ∃ p : ℝ × ℝ, p ∈ Circle ∩ XAxis ∧ p ≠ Origin ∧ p.1 = 10 := by
  sorry

end circle_x_axis_intersection_l1126_112678


namespace nested_radical_equation_l1126_112613

theorem nested_radical_equation (x : ℝ) : 
  x = 34 → 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 := by
sorry

end nested_radical_equation_l1126_112613


namespace arithmetic_mean_after_removal_l1126_112629

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 100 →
  x = 50 →
  y = 60 →
  x ∈ S →
  y ∈ S →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2)) = 44.8 := by
  sorry

end arithmetic_mean_after_removal_l1126_112629


namespace parabola_decreasing_condition_l1126_112685

/-- Represents a parabola of the form y = -5(x + m)² - 3 -/
def Parabola (m : ℝ) : ℝ → ℝ := λ x ↦ -5 * (x + m)^2 - 3

/-- States that the parabola is decreasing for x ≥ 2 -/
def IsDecreasingForXGeq2 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ 2 → x₂ ≥ 2 → x₁ < x₂ → Parabola m x₁ > Parabola m x₂

theorem parabola_decreasing_condition (m : ℝ) :
  IsDecreasingForXGeq2 m → m ≥ -2 := by
  sorry

end parabola_decreasing_condition_l1126_112685


namespace product_mod_twenty_l1126_112625

theorem product_mod_twenty : (93 * 68 * 105) % 20 = 0 := by
  sorry

end product_mod_twenty_l1126_112625


namespace parallel_lines_c_value_l1126_112612

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 6x + 3 and y = 3cx + 1 are parallel -/
theorem parallel_lines_c_value : 
  (∀ x y : ℝ, y = 6 * x + 3 ↔ y = 3 * c * x + 1) → c = 2 := by
  sorry

end parallel_lines_c_value_l1126_112612


namespace tan_fifteen_degree_fraction_l1126_112669

theorem tan_fifteen_degree_fraction : 
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end tan_fifteen_degree_fraction_l1126_112669


namespace nathan_added_half_blankets_l1126_112675

/-- The fraction of blankets Nathan added to his bed -/
def blanket_fraction (total_blankets : ℕ) (temp_per_blanket : ℕ) (total_temp_increase : ℕ) : ℚ :=
  (total_temp_increase / temp_per_blanket : ℚ) / total_blankets

/-- Theorem stating that Nathan added 1/2 of his blankets -/
theorem nathan_added_half_blankets :
  blanket_fraction 14 3 21 = 1/2 := by
  sorry

end nathan_added_half_blankets_l1126_112675


namespace estimate_passing_papers_l1126_112628

/-- Estimates the number of passing papers in a population based on a sample --/
theorem estimate_passing_papers 
  (total_papers : ℕ) 
  (sample_size : ℕ) 
  (passing_in_sample : ℕ) 
  (h1 : total_papers = 10000)
  (h2 : sample_size = 500)
  (h3 : passing_in_sample = 420) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 8400 :=
sorry

end estimate_passing_papers_l1126_112628


namespace probability_second_draw_given_first_l1126_112636

/-- The probability of drawing a high-quality item on the second draw, given that the first draw was a high-quality item, when there are 5 high-quality items and 3 defective items in total. -/
theorem probability_second_draw_given_first (total_items : ℕ) (high_quality : ℕ) (defective : ℕ) :
  total_items = high_quality + defective →
  high_quality = 5 →
  defective = 3 →
  (high_quality - 1 : ℚ) / (total_items - 1) = 4 / 7 := by
  sorry

end probability_second_draw_given_first_l1126_112636


namespace sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l1126_112654

theorem sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023 :
  Real.sqrt 4 - |Real.sqrt 3 - 2| + (-1)^2023 = Real.sqrt 3 - 1 := by
  sorry

end sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l1126_112654


namespace function_properties_l1126_112604

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f e x ≥ f e x₀) ∧
  (∀ M : ℝ, ∃ x > 0, f e x > M) ∧
  -- Part 2
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1/a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ x : ℝ, x > 0 → Real.exp (2*x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry

end function_properties_l1126_112604


namespace binomial_expansion_sum_zero_l1126_112649

theorem binomial_expansion_sum_zero (n k : ℕ) (b : ℝ) 
  (h1 : n ≥ 2)
  (h2 : b ≠ 0)
  (h3 : k > 0)
  (h4 : n.choose 1 * b^(n-1) * (k+1) + n.choose 2 * b^(n-2) * (k+1)^2 = 0) :
  n = 2*k + 2 := by
sorry

end binomial_expansion_sum_zero_l1126_112649


namespace points_scored_in_quarter_l1126_112696

/-- Calculates the total points scored in a basketball quarter -/
def total_points_scored (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Proves that given four 2-point shots and two 3-point shots, the total points scored is 14 -/
theorem points_scored_in_quarter : total_points_scored 4 2 = 14 := by
  sorry

end points_scored_in_quarter_l1126_112696


namespace triangle_obtuse_l1126_112699

theorem triangle_obtuse (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  π / 2 < C ∧ C < π :=
sorry

end triangle_obtuse_l1126_112699


namespace robot_price_ratio_l1126_112671

/-- The ratio of the price Tom should pay to the original price of the robot -/
theorem robot_price_ratio (original_price tom_price : ℚ) 
  (h1 : original_price = 3)
  (h2 : tom_price = 9) :
  tom_price / original_price = 3 := by
sorry

end robot_price_ratio_l1126_112671


namespace point_on_or_outside_circle_l1126_112639

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (a, 2 - a)

-- Theorem statement
theorem point_on_or_outside_circle : 
  ∀ a : ℝ, (P a) ∈ C ∨ (P a) ∉ interior C :=
sorry

end point_on_or_outside_circle_l1126_112639


namespace exam_mean_score_l1126_112691

/-- Given a distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean of the
    distribution is 76. -/
theorem exam_mean_score (μ σ : ℝ)
    (below_mean : μ - 2 * σ = 60)
    (above_mean : μ + 3 * σ = 100) :
    μ = 76 := by
  sorry


end exam_mean_score_l1126_112691


namespace least_positive_slope_line_l1126_112665

/-- The curve equation -/
def curve (x y : ℝ) : Prop := 4 * x^2 - y^2 - 8 * x = 12

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - m

/-- The line contains the point (1, 0) -/
def contains_point (m : ℝ) : Prop := line m 1 0

/-- The line does not intersect the curve -/
def no_intersection (m : ℝ) : Prop := ∀ x y : ℝ, line m x y → ¬curve x y

/-- The slope is positive -/
def positive_slope (m : ℝ) : Prop := m > 0

theorem least_positive_slope_line :
  ∃ m : ℝ, m = 2 ∧
    contains_point m ∧
    no_intersection m ∧
    positive_slope m ∧
    ∀ m' : ℝ, m' ≠ m → contains_point m' → no_intersection m' → positive_slope m' → m' > m :=
sorry

end least_positive_slope_line_l1126_112665


namespace five_objects_three_boxes_l1126_112687

/-- Number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is equal to 150 -/
theorem five_objects_three_boxes : distributionCount 5 3 = 150 := by sorry

end five_objects_three_boxes_l1126_112687


namespace square_condition_implies_value_l1126_112668

theorem square_condition_implies_value (k : ℕ) :
  (∃ m n : ℕ, (4 * k + 5 = m^2) ∧ (9 * k + 4 = n^2)) →
  (7 * k + 4 = 39) := by
sorry

end square_condition_implies_value_l1126_112668


namespace nonnegative_solutions_count_l1126_112602

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 :=
by sorry

end nonnegative_solutions_count_l1126_112602


namespace sum_of_powers_l1126_112657

theorem sum_of_powers (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2 → y = 4 * a ^ 2 - a := by
  sorry

end sum_of_powers_l1126_112657


namespace fourth_root_power_eight_l1126_112662

theorem fourth_root_power_eight : (2^6)^(1/4)^8 = 4096 := by
  sorry

end fourth_root_power_eight_l1126_112662


namespace parabola_points_theorem_l1126_112680

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions
def satisfies_conditions (A B : Point) : Prop :=
  A.y = parabola A.x ∧ 
  B.y = parabola B.x ∧ 
  A.x < 0 ∧ 
  B.x > 0 ∧
  A.y > B.y

-- Define the theorem
theorem parabola_points_theorem (A B : Point) (h : satisfies_conditions A B) :
  (A.x = -4 ∧ B.x = 2) ∨ (A.x = 4 ∧ B.x = -2) :=
sorry

end parabola_points_theorem_l1126_112680


namespace polynomial_roots_l1126_112655

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem polynomial_roots (P : ℝ → ℝ) (h_nonzero : P ≠ 0) 
  (h_form : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) :
  (∃ c ≠ 0, ∀ x, P x = c * (x^2 - x - 1)) ∧
  (∀ x, P x = 0 ↔ x = golden_ratio ∨ x = 1 - golden_ratio) :=
sorry

end polynomial_roots_l1126_112655


namespace gcd_1617_1225_l1126_112620

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 := by sorry

end gcd_1617_1225_l1126_112620


namespace arrange_balls_count_l1126_112660

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeColoredBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.choose (red + yellow + white) white *
  Nat.choose (red + yellow) red *
  Nat.choose yellow yellow

/-- Theorem stating that arranging 2 red, 3 yellow, and 4 white balls results in 1260 arrangements -/
theorem arrange_balls_count : arrangeColoredBalls 2 3 4 = 1260 := by
  sorry

end arrange_balls_count_l1126_112660


namespace no_convincing_statement_when_guilty_l1126_112697

/-- Represents a statement made in court -/
def Statement : Type := String

/-- Represents the state of being guilty or innocent -/
inductive GuiltState
| Guilty
| Innocent

/-- Represents a jury's belief about guilt -/
inductive JuryBelief
| BelievesGuilty
| BelievesInnocent

/-- A function that models how a rational jury processes a statement -/
def rationalJuryProcess : Statement → GuiltState → JuryBelief := sorry

/-- The theorem stating that it's impossible to convince a rational jury of innocence when guilty -/
theorem no_convincing_statement_when_guilty :
  ∀ (s : Statement), rationalJuryProcess s GuiltState.Guilty ≠ JuryBelief.BelievesInnocent := by
  sorry

end no_convincing_statement_when_guilty_l1126_112697


namespace largest_x_sqrt_3x_eq_5x_l1126_112611

theorem largest_x_sqrt_3x_eq_5x :
  (∃ x : ℝ, x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ x : ℝ, Real.sqrt (3 * x) = 5 * x → x ≤ 3/25) ∧
  Real.sqrt (3 * (3/25)) = 5 * (3/25) :=
by sorry

end largest_x_sqrt_3x_eq_5x_l1126_112611


namespace marble_prism_weight_l1126_112616

/-- Calculates the weight of a rectangular prism with a square base -/
def weight_rectangular_prism (height : ℝ) (base_side : ℝ) (density : ℝ) : ℝ :=
  height * base_side * base_side * density

/-- Proves that the weight of the given marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  weight_rectangular_prism 8 2 2700 = 86400 := by
  sorry

end marble_prism_weight_l1126_112616


namespace gcd_lcm_sum_8_12_l1126_112647

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end gcd_lcm_sum_8_12_l1126_112647


namespace smallest_n_same_divisors_l1126_112656

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Checks if three consecutive natural numbers have the same number of divisors -/
def same_num_divisors (n : ℕ) : Prop :=
  num_divisors n = num_divisors (n + 1) ∧ num_divisors n = num_divisors (n + 2)

/-- 33 is the smallest natural number n such that n, n+1, and n+2 have the same number of divisors -/
theorem smallest_n_same_divisors :
  (∀ m : ℕ, m < 33 → ¬ same_num_divisors m) ∧ same_num_divisors 33 := by
  sorry

end smallest_n_same_divisors_l1126_112656


namespace quadratic_equation_equivalence_l1126_112686

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
sorry

end quadratic_equation_equivalence_l1126_112686


namespace copy_machines_output_l1126_112689

/-- The rate of the first copy machine in copies per minute -/
def rate1 : ℕ := 30

/-- The rate of the second copy machine in copies per minute -/
def rate2 : ℕ := 55

/-- The time period in minutes -/
def time : ℕ := 30

/-- The total number of copies made by both machines in the given time period -/
def total_copies : ℕ := rate1 * time + rate2 * time

theorem copy_machines_output : total_copies = 2550 := by
  sorry

end copy_machines_output_l1126_112689


namespace smallest_positive_multiple_of_45_l1126_112638

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end smallest_positive_multiple_of_45_l1126_112638


namespace point_on_x_axis_l1126_112667

/-- A point P with coordinates (m+3, m-2) lies on the x-axis if and only if its coordinates are (5,0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 2 = 0 ∧ (m + 3, m - 2).1 = m + 3 ∧ (m + 3, m - 2).2 = m - 2) ↔ 
  (m + 3, m - 2) = (5, 0) :=
by sorry

end point_on_x_axis_l1126_112667


namespace sin_transformation_l1126_112644

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x - π / 12) + π / 2) := by
  sorry

end sin_transformation_l1126_112644


namespace sum_of_decimals_l1126_112673

theorem sum_of_decimals : 
  let addend1 : ℚ := 57/100
  let addend2 : ℚ := 23/100
  addend1 + addend2 = 4/5 :=
by sorry

end sum_of_decimals_l1126_112673


namespace instantaneous_velocity_at_2_l1126_112618

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 4 := by
  sorry

end instantaneous_velocity_at_2_l1126_112618


namespace contractor_daily_wage_l1126_112688

theorem contractor_daily_wage
  (total_days : ℕ)
  (absence_fine : ℚ)
  (total_payment : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : absence_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10)
  : ∃ (daily_wage : ℚ), daily_wage = 25 ∧
    (total_days - absent_days) * daily_wage - absent_days * absence_fine = total_payment :=
by
  sorry

end contractor_daily_wage_l1126_112688


namespace bracelet_capacity_is_fifteen_l1126_112652

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ
  total_fill_cost : ℕ

/-- Calculates the bracelet display capacity given the store's inventory and pricing --/
def bracelet_display_capacity (store : JewelryStore) : ℕ :=
  store.current_bracelets +
    ((store.total_fill_cost -
      (store.necklace_price * (store.necklace_capacity - store.current_necklaces) +
       store.ring_price * (store.ring_capacity - store.current_rings)))
     / store.bracelet_price)

/-- Theorem stating that for the given store configuration, the bracelet display capacity is 15 --/
theorem bracelet_capacity_is_fifteen :
  let store : JewelryStore := {
    necklace_capacity := 12,
    current_necklaces := 5,
    ring_capacity := 30,
    current_rings := 18,
    current_bracelets := 8,
    necklace_price := 4,
    ring_price := 10,
    bracelet_price := 5,
    total_fill_cost := 183
  }
  bracelet_display_capacity store = 15 := by
  sorry


end bracelet_capacity_is_fifteen_l1126_112652


namespace valid_parameterization_l1126_112695

/-- Defines a line in 2D space --/
structure Line2D where
  slope : ℚ
  intercept : ℚ

/-- Defines a vector parameterization of a line --/
structure VectorParam where
  x₀ : ℚ
  y₀ : ℚ
  a : ℚ
  b : ℚ

/-- Checks if a vector parameterization is valid for a given line --/
def isValidParam (l : Line2D) (p : VectorParam) : Prop :=
  ∃ (k : ℚ), p.a = k * 5 ∧ p.b = k * 7 ∧ 
  p.y₀ = l.slope * p.x₀ + l.intercept

/-- The main theorem to prove --/
theorem valid_parameterization (l : Line2D) (p : VectorParam) :
  l.slope = 7/5 ∧ l.intercept = -23/5 →
  isValidParam l p ↔ 
    (p.x₀ = 5 ∧ p.y₀ = 2 ∧ p.a = -5 ∧ p.b = -7) ∨
    (p.x₀ = 23 ∧ p.y₀ = 7 ∧ p.a = 10 ∧ p.b = 14) ∨
    (p.x₀ = 3 ∧ p.y₀ = -8/5 ∧ p.a = 7/5 ∧ p.b = 1) ∨
    (p.x₀ = 0 ∧ p.y₀ = -23/5 ∧ p.a = 25 ∧ p.b = -35) :=
by sorry

end valid_parameterization_l1126_112695


namespace lattice_point_in_triangle_l1126_112609

/-- A point in a 2D integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral in a 2D integer lattice -/
structure ConvexLatticeQuadrilateral where
  P : LatticePoint
  Q : LatticePoint
  R : LatticePoint
  S : LatticePoint
  is_convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The angle between two vectors -/
def angle (v1 v2 : LatticePoint → LatticePoint) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def is_in_triangle (X P Q E : LatticePoint) : Prop := sorry

theorem lattice_point_in_triangle
  (PQRS : ConvexLatticeQuadrilateral)
  (E : LatticePoint)
  (h_diagonals_intersect : E = sorry)  -- E is the intersection of diagonals
  (h_angle_sum : angle (λ p => PQRS.P) (λ p => PQRS.Q) < 180) :
  ∃ X : LatticePoint, X ≠ PQRS.P ∧ X ≠ PQRS.Q ∧ is_in_triangle X PQRS.P PQRS.Q E :=
sorry

end lattice_point_in_triangle_l1126_112609


namespace max_value_sum_of_roots_l1126_112643

theorem max_value_sum_of_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end max_value_sum_of_roots_l1126_112643


namespace probability_sum_12_l1126_112633

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The target sum we're aiming for -/
def targetSum : ℕ := 12

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 12) -/
def favorableOutcomes : ℕ := 10

/-- The probability of rolling a sum of 12 with three standard six-sided dice -/
theorem probability_sum_12 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 10 / 216 := by sorry

end probability_sum_12_l1126_112633


namespace coins_value_is_78_percent_of_dollar_l1126_112653

-- Define the value of each coin in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of each coin
def num_pennies : ℕ := 3
def num_nickels : ℕ := 2
def num_dimes : ℕ := 4
def num_quarters : ℕ := 1

-- Define the total value in cents
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

-- Define one dollar in cents
def dollar_in_cents : ℕ := 100

-- Theorem to prove
theorem coins_value_is_78_percent_of_dollar : 
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 78 / 100 := by
  sorry

end coins_value_is_78_percent_of_dollar_l1126_112653


namespace min_pages_per_day_l1126_112627

theorem min_pages_per_day (total_pages : ℕ) (days_in_week : ℕ) : 
  total_pages = 220 → days_in_week = 7 → 
  ∃ (min_pages : ℕ), 
    min_pages * days_in_week ≥ total_pages ∧ 
    ∀ (x : ℕ), x * days_in_week ≥ total_pages → x ≥ min_pages ∧
    min_pages = 32 := by
  sorry

end min_pages_per_day_l1126_112627


namespace total_cost_after_rebate_l1126_112617

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_price : ℕ := 83
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12

theorem total_cost_after_rebate :
  (polo_shirt_price * polo_shirt_quantity +
   necklace_price * necklace_quantity +
   computer_game_price * computer_game_quantity) - rebate = 322 := by
  sorry

end total_cost_after_rebate_l1126_112617


namespace hyperbola_equation_l1126_112626

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3) →
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end hyperbola_equation_l1126_112626


namespace distance_between_points_l1126_112677

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
sorry

end distance_between_points_l1126_112677


namespace claire_age_in_two_years_l1126_112693

/-- Given that Jessica is 24 years old and 6 years older than Claire, 
    prove that Claire will be 20 years old in two years. -/
theorem claire_age_in_two_years 
  (jessica_age : ℕ) 
  (claire_age : ℕ) 
  (h1 : jessica_age = 24)
  (h2 : jessica_age = claire_age + 6) : 
  claire_age + 2 = 20 := by
sorry

end claire_age_in_two_years_l1126_112693


namespace complex_absolute_value_product_l1126_112624

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 18 * Real.sqrt 2 := by
  sorry

end complex_absolute_value_product_l1126_112624


namespace circle_tangent_to_three_lines_l1126_112634

-- Define the types for lines and circles
variable (Line Circle : Type)

-- Define the tangent relation between a circle and a line
variable (tangent_to : Circle → Line → Prop)

-- Define the intersection angle between two lines
variable (intersection_angle : Line → Line → ℝ)

-- Define the main theorem
theorem circle_tangent_to_three_lines 
  (C : Circle) (l m n : Line) :
  (tangent_to C l ∧ tangent_to C m ∧ tangent_to C n) →
  (∃ (C' : Circle), 
    tangent_to C' l ∧ tangent_to C' m ∧ tangent_to C' n) ∧
  (intersection_angle l m = π/3 ∧ 
   intersection_angle m n = π/3 ∧ 
   intersection_angle n l = π/3) :=
by sorry

end circle_tangent_to_three_lines_l1126_112634
