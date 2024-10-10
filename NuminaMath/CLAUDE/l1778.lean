import Mathlib

namespace cell_division_after_three_hours_l1778_177819

/-- Represents the number of cells after a given number of 30-minute periods -/
def cells_after_periods (n : ℕ) : ℕ := 2^n

/-- Represents the number of 30-minute periods in a given number of hours -/
def periods_in_hours (hours : ℕ) : ℕ := 2 * hours

theorem cell_division_after_three_hours :
  cells_after_periods (periods_in_hours 3) = 64 := by
  sorry

#eval cells_after_periods (periods_in_hours 3)

end cell_division_after_three_hours_l1778_177819


namespace min_relevant_number_l1778_177871

def A (n : ℕ) := Finset.range (2*n + 1) \ {0}

def is_relevant_number (n m : ℕ) : Prop :=
  n ≥ 2 ∧ m ≥ 4 ∧
  ∀ (P : Finset ℕ), P ⊆ A n → P.card = m →
    ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem min_relevant_number (n : ℕ) :
  n ≥ 2 → (∃ (m : ℕ), is_relevant_number n m) →
  ∃ (m : ℕ), is_relevant_number n m ∧ ∀ (k : ℕ), is_relevant_number n k → m ≤ k :=
by sorry

end min_relevant_number_l1778_177871


namespace max_product_under_constraint_l1778_177832

theorem max_product_under_constraint (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (constraint1 : x * y * z + w = (x + w) * (y + w) * (z + w))
  (constraint2 : x + y + z + w = 1) : 
  x * y * z * w ≤ 1 / 256 := by
  sorry

end max_product_under_constraint_l1778_177832


namespace room_width_proof_l1778_177880

/-- Proves that a rectangular room with given dimensions has a specific width -/
theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 19 →
  veranda_width = 2 →
  veranda_area = 140 →
  ∃ (room_width : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end room_width_proof_l1778_177880


namespace correct_matching_probability_l1778_177844

theorem correct_matching_probability (n : ℕ) (hn : n = 4) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 24 :=
sorry

end correct_matching_probability_l1778_177844


namespace triangle_side_calculation_l1778_177895

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  a = Real.sqrt 3 →
  Real.sin B = 1/2 →
  C = π/6 →
  -- Conclusion
  b = 1 := by
  sorry

end triangle_side_calculation_l1778_177895


namespace airplane_seats_l1778_177831

theorem airplane_seats : ∃ (s : ℝ), 
  (30 : ℝ) + 0.2 * s + 0.75 * s = s ∧ s = 600 := by
  sorry

end airplane_seats_l1778_177831


namespace convex_set_enclosure_l1778_177808

-- Define a convex set in 2D space
variable (Φ : Set (ℝ × ℝ))

-- Define the property of being convex
def IsConvex (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of being centrally symmetric
def IsCentrallySymmetric (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of one set enclosing another
def Encloses (S T : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set
noncomputable def Area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle
def IsTriangle (S : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem convex_set_enclosure (h : IsConvex Φ) : 
  ∃ S : Set (ℝ × ℝ), 
    IsConvex S ∧ 
    IsCentrallySymmetric S ∧ 
    Encloses S Φ ∧ 
    Area S ≤ 2 * Area Φ ∧
    (IsTriangle Φ → Area S ≥ 2 * Area Φ) := by
  sorry

end convex_set_enclosure_l1778_177808


namespace number_problem_l1778_177834

theorem number_problem (x : ℝ) : 50 + 5 * 12 / (x / 3) = 51 → x = 180 := by
  sorry

end number_problem_l1778_177834


namespace remainder_x_105_divided_by_x_plus_1_4_l1778_177864

theorem remainder_x_105_divided_by_x_plus_1_4 (x : ℤ) :
  x^105 ≡ 195300*x^3 + 580440*x^2 + 576085*x + 189944 [ZMOD (x + 1)^4] := by
  sorry

end remainder_x_105_divided_by_x_plus_1_4_l1778_177864


namespace binomial_n_n_minus_3_l1778_177846

theorem binomial_n_n_minus_3 (n : ℕ) (h : n ≥ 3) :
  Nat.choose n (n - 3) = n * (n - 1) * (n - 2) / 6 := by
  sorry

end binomial_n_n_minus_3_l1778_177846


namespace trapezoid_construction_possible_l1778_177898

/-- Represents a trapezoid with sides a, b, c, d and diagonals d₁, d₂ -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_parallel : c = d
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0
  h_inequality₁ : d₁ - d₂ < a + b
  h_inequality₂ : a + b < d₁ + d₂

/-- A trapezoid can be constructed given parallel sides and diagonals satisfying certain conditions -/
theorem trapezoid_construction_possible (a b c d d₁ d₂ : ℝ) 
  (h_parallel : c = d)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0)
  (h_inequality₁ : d₁ - d₂ < a + b)
  (h_inequality₂ : a + b < d₁ + d₂) :
  ∃ t : Trapezoid, t.a = a ∧ t.b = b ∧ t.c = c ∧ t.d = d ∧ t.d₁ = d₁ ∧ t.d₂ = d₂ :=
by sorry

end trapezoid_construction_possible_l1778_177898


namespace orchid_bushes_total_park_orchid_bushes_l1778_177843

/-- The total number of orchid bushes after planting is equal to the sum of the current number of bushes and the number of bushes planted over two days. -/
theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current + today + tomorrow = current + today + tomorrow :=
by sorry

/-- Given the specific numbers from the problem -/
theorem park_orchid_bushes :
  let current : ℕ := 47
  let today : ℕ := 37
  let tomorrow : ℕ := 25
  current + today + tomorrow = 109 :=
by sorry

end orchid_bushes_total_park_orchid_bushes_l1778_177843


namespace gcd_of_container_volumes_l1778_177816

theorem gcd_of_container_volumes : Nat.gcd 496 (Nat.gcd 403 (Nat.gcd 713 (Nat.gcd 824 1171))) = 1 := by
  sorry

end gcd_of_container_volumes_l1778_177816


namespace equation_one_solution_equation_two_solution_l1778_177825

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  (1 / 3) * x^2 = 2 ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  8 * (x - 1)^3 = -(27 / 8) ↔ x = 1 / 4 := by sorry

end equation_one_solution_equation_two_solution_l1778_177825


namespace total_books_on_shelves_l1778_177812

/-- The number of book shelves -/
def num_shelves : ℕ := 150

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 15

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_on_shelves :
  total_books = 2250 :=
by sorry

end total_books_on_shelves_l1778_177812


namespace camping_group_solution_l1778_177818

/-- Represents the camping group -/
structure CampingGroup where
  initialTotal : ℕ
  initialGirls : ℕ

/-- Conditions of the camping group problem -/
class CampingGroupProblem (g : CampingGroup) where
  initial_ratio : g.initialGirls = g.initialTotal / 2
  final_ratio : (g.initialGirls + 1) * 10 = 6 * (g.initialTotal - 2)

/-- The theorem stating the solution to the camping group problem -/
theorem camping_group_solution (g : CampingGroup) [CampingGroupProblem g] : 
  g.initialGirls = 11 := by
  sorry

#check camping_group_solution

end camping_group_solution_l1778_177818


namespace cookie_problem_l1778_177897

theorem cookie_problem (frank mike millie : ℕ) : 
  frank = (mike / 2) - 3 →
  mike = 3 * millie →
  frank = 3 →
  millie = 4 := by
sorry

end cookie_problem_l1778_177897


namespace second_number_value_l1778_177855

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 157.5)
  (ratio_AB : A / B = 3.5 / 4.25)
  (ratio_BC : B / C = 7.5 / 11.25)
  (diff_AC : A - C = 12.75) :
  B = 18.75 := by
sorry

end second_number_value_l1778_177855


namespace enclosed_area_theorem_l1778_177851

noncomputable def g (x : ℝ) : ℝ := 2 - Real.sqrt (1 - (2*x/3)^2)

def domain : Set ℝ := Set.Icc (-3/2) (3/2)

theorem enclosed_area_theorem (A : ℝ) :
  A = 2 * (π * (3/2)^2 / 2 - ∫ x in (Set.Icc 0 (3/2)), g x) :=
sorry

end enclosed_area_theorem_l1778_177851


namespace power_multiplication_l1778_177840

theorem power_multiplication (n : ℕ) : n * (n^(n - 1)) = n^n :=
by
  sorry

#check power_multiplication 3000

end power_multiplication_l1778_177840


namespace red_probability_both_jars_l1778_177878

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Calculates the probability of drawing a red button from a jar -/
def Jar.redProbability (j : Jar) : ℚ := j.red / j.total

/-- Represents the initial state of Jar A -/
def initialJarA : Jar := { red := 8, blue := 8 }

/-- Represents the transfer process -/
def transfer (j : Jar) : (Jar × Jar) :=
  let redTransfer := j.red / 3
  let blueTransfer := j.blue / 2
  let newJarA : Jar := { red := j.red - redTransfer, blue := j.blue - blueTransfer }
  let jarB : Jar := { red := redTransfer, blue := blueTransfer }
  (newJarA, jarB)

/-- The main theorem stating the probability of drawing red buttons from both jars -/
theorem red_probability_both_jars :
  let (jarA, jarB) := transfer initialJarA
  (jarA.redProbability * jarB.redProbability) = 5 / 21 := by
  sorry


end red_probability_both_jars_l1778_177878


namespace arithmetic_sequence_proof_l1778_177890

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a (n : ℕ) : ℝ := S n - S (n-1)

theorem arithmetic_sequence_proof :
  ∃ (d : ℝ), ∀ (n : ℕ), n ≥ 1 → a n = a 1 + (n - 1) * d :=
sorry

end arithmetic_sequence_proof_l1778_177890


namespace election_votes_calculation_l1778_177803

theorem election_votes_calculation (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes / 2 + majority : ℚ) →
  total_votes = 6500 := by
  sorry

end election_votes_calculation_l1778_177803


namespace divided_right_triangle_area_ratio_l1778_177879

/-- A right triangle divided by lines parallel to its legs through a point on its hypotenuse -/
structure DividedRightTriangle where
  /-- The area of the square formed by the division -/
  square_area : ℝ
  /-- The area of the first small right triangle -/
  small_triangle1_area : ℝ
  /-- The area of the second small right triangle -/
  small_triangle2_area : ℝ
  /-- The square_area is positive -/
  square_area_pos : 0 < square_area

/-- The theorem stating the relationship between the areas -/
theorem divided_right_triangle_area_ratio
  (t : DividedRightTriangle)
  (m : ℝ)
  (h : t.small_triangle1_area = m * t.square_area) :
  t.small_triangle2_area = (1 / (4 * m)) * t.square_area :=
by sorry

end divided_right_triangle_area_ratio_l1778_177879


namespace absolute_value_sum_lower_bound_l1778_177806

theorem absolute_value_sum_lower_bound :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x + 2| + |x - 1| < 3 + ε) :=
sorry

end absolute_value_sum_lower_bound_l1778_177806


namespace man_son_age_difference_l1778_177872

/-- The age difference between a man and his son -/
def age_difference : ℕ → ℕ → ℕ
  | father_age, son_age => father_age - son_age

/-- Theorem stating the age difference between the man and his son -/
theorem man_son_age_difference :
  ∀ (man_age son_age : ℕ),
    son_age = 44 →
    man_age + 2 = 2 * (son_age + 2) →
    age_difference man_age son_age = 46 := by
  sorry

#check man_son_age_difference

end man_son_age_difference_l1778_177872


namespace tangent_circles_diameter_intersection_l1778_177892

/-- Given three circles that are pairwise tangent, the lines connecting
    the tangency points of two circles intersect the third circle at
    the endpoints of its diameter. -/
theorem tangent_circles_diameter_intersection
  (O₁ O₂ O₃ : ℝ × ℝ) -- Centers of the three circles
  (r₁ r₂ r₃ : ℝ) -- Radii of the three circles
  (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) -- Radii are positive
  (h_tangent : -- Circles are pairwise tangent
    (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 = (r₁ + r₂)^2 ∧
    (O₂.1 - O₃.1)^2 + (O₂.2 - O₃.2)^2 = (r₂ + r₃)^2 ∧
    (O₃.1 - O₁.1)^2 + (O₃.2 - O₁.2)^2 = (r₃ + r₁)^2)
  (h_distinct : O₁ ≠ O₂ ∧ O₂ ≠ O₃ ∧ O₃ ≠ O₁) -- Centers are distinct
  : ∃ (A B C : ℝ × ℝ), -- Tangency points
    -- A is on circle 1 and 2
    ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₁^2 ∧ (A.1 - O₂.1)^2 + (A.2 - O₂.2)^2 = r₂^2) ∧
    -- B is on circle 2 and 3
    ((B.1 - O₂.1)^2 + (B.2 - O₂.2)^2 = r₂^2 ∧ (B.1 - O₃.1)^2 + (B.2 - O₃.2)^2 = r₃^2) ∧
    -- C is on circle 1 and 3
    ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ (C.1 - O₃.1)^2 + (C.2 - O₃.2)^2 = r₃^2) ∧
    -- Lines AB and AC intersect circle 3 at diameter endpoints
    ∃ (M K : ℝ × ℝ),
      (M.1 - O₃.1)^2 + (M.2 - O₃.2)^2 = r₃^2 ∧
      (K.1 - O₃.1)^2 + (K.2 - O₃.2)^2 = r₃^2 ∧
      (M.1 - K.1)^2 + (M.2 - K.2)^2 = 4 * r₃^2 ∧
      (∃ t : ℝ, M = (1 - t) • A + t • B) ∧
      (∃ s : ℝ, K = (1 - s) • A + s • C) := by
  sorry

end tangent_circles_diameter_intersection_l1778_177892


namespace cos_300_deg_l1778_177849

/-- Cosine of 300 degrees is equal to 1/2 -/
theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_deg_l1778_177849


namespace floor_sqrt_80_l1778_177888

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l1778_177888


namespace expression_simplification_l1778_177858

theorem expression_simplification (a b c : ℝ) :
  3 / 4 * (6 * a^2 - 12 * a) - 8 / 5 * (3 * b^2 + 15 * b) + (2 * c^2 - 6 * c) / 6 =
  (9/2) * a^2 - 9 * a - (24/5) * b^2 - 24 * b + (1/3) * c^2 - c :=
by sorry

end expression_simplification_l1778_177858


namespace incorrect_proposition_statement_l1778_177893

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end incorrect_proposition_statement_l1778_177893


namespace least_addition_for_divisibility_l1778_177860

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 4 ∧ 
  (∀ (y : ℕ), (1100 + y) % 23 = 0 → y ≥ x) ∧ 
  (1100 + x) % 23 = 0 := by
  sorry

end least_addition_for_divisibility_l1778_177860


namespace parabola_point_x_coordinate_l1778_177868

theorem parabola_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h1 : (P.2)^2 = 4 * P.1) 
  (h2 : Real.sqrt ((P.1 - 1)^2 + P.2^2) = 10) : 
  P.1 = 9 := by
sorry

end parabola_point_x_coordinate_l1778_177868


namespace money_division_l1778_177883

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 15 →
  a + b + c = 540 :=
by
  sorry

end money_division_l1778_177883


namespace circular_fields_radius_l1778_177862

theorem circular_fields_radius (r₁ r₂ : ℝ) : 
  r₂ = 10 →
  π * r₁^2 = 0.09 * (π * r₂^2) →
  r₁ = 3 := by
sorry

end circular_fields_radius_l1778_177862


namespace cricket_team_right_handed_count_l1778_177826

/-- Calculates the number of right-handed players in a cricket team -/
def right_handed_players (total_players throwers : ℕ) : ℕ :=
  let non_throwers := total_players - throwers
  let left_handed_non_throwers := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed_non_throwers
  throwers + right_handed_non_throwers

theorem cricket_team_right_handed_count :
  right_handed_players 70 37 = 59 := by
  sorry

end cricket_team_right_handed_count_l1778_177826


namespace marble_fraction_after_change_l1778_177887

theorem marble_fraction_after_change (total : ℚ) (h : total > 0) :
  let initial_blue := (2 / 3 : ℚ) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by
  sorry

end marble_fraction_after_change_l1778_177887


namespace x_value_l1778_177875

theorem x_value (x : ℝ) : x = 150 * (1 + 0.75) → x = 262.5 := by
  sorry

end x_value_l1778_177875


namespace evaluate_expression_l1778_177854

theorem evaluate_expression (x z : ℝ) (hx : x = 5) (hz : z = 4) :
  z^2 * (z^2 - 4*x) = -64 := by
  sorry

end evaluate_expression_l1778_177854


namespace lcm_14_21_35_l1778_177823

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by sorry

end lcm_14_21_35_l1778_177823


namespace circle_area_and_diameter_l1778_177800

theorem circle_area_and_diameter (C : ℝ) (h : C = 18 * Real.pi) : ∃ (A d : ℝ), A = 81 * Real.pi ∧ d = 18 ∧ A = Real.pi * (d / 2)^2 ∧ C = Real.pi * d := by
  sorry

end circle_area_and_diameter_l1778_177800


namespace ampersand_composition_l1778_177889

-- Define the & operation
def ampersand_right (y : ℤ) : ℤ := 9 - y

-- Define the & operation
def ampersand_left (y : ℤ) : ℤ := y - 9

-- Theorem to prove
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end ampersand_composition_l1778_177889


namespace g_of_5_l1778_177873

def g (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 30*x^3 - 45*x^2 + 24*x + 50

theorem g_of_5 : g 5 = 2795 := by sorry

end g_of_5_l1778_177873


namespace library_books_count_l1778_177805

/-- The number of shelves in the library -/
def num_shelves : ℕ := 1780

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_books_count : total_books = 14240 := by
  sorry

end library_books_count_l1778_177805


namespace two_common_tangents_l1778_177817

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y r : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = r^2

/-- Theorem stating the condition for exactly two common tangent lines -/
theorem two_common_tangents (r : ℝ) :
  (r > 0) →
  (∃ (x y : ℝ), C₁ x y ∧ C₂ x y r) ↔ (Real.sqrt 5 - 2 < r ∧ r < Real.sqrt 5 + 2) :=
sorry

end two_common_tangents_l1778_177817


namespace circle_area_equality_l1778_177859

theorem circle_area_equality (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area := θ * r^2
  let triangle_area := (r^2 * Real.tan θ * Real.tan (2 * θ)) / 2
  let circle_area := π * r^2
  triangle_area = circle_area - sector_area ↔ 2 * θ = Real.tan θ * Real.tan (2 * θ) :=
by sorry

end circle_area_equality_l1778_177859


namespace meeting_attendees_l1778_177852

/-- The number of handshakes in a meeting where every two people shake hands. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 12 people in the meeting given the conditions. -/
theorem meeting_attendees : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end meeting_attendees_l1778_177852


namespace original_list_size_l1778_177884

/-- Given a list of integers, if appending 25 increases the mean by 3,
    and then appending -4 decreases the mean by 1.5,
    prove that the original list contained 4 integers. -/
theorem original_list_size (l : List Int) : 
  (((l.sum + 25) / (l.length + 1) : ℚ) = (l.sum / l.length : ℚ) + 3) →
  (((l.sum + 21) / (l.length + 2) : ℚ) = (l.sum / l.length : ℚ) + 1.5) →
  l.length = 4 := by
sorry


end original_list_size_l1778_177884


namespace similar_right_triangles_l1778_177811

theorem similar_right_triangles (y : ℝ) : 
  y > 0 →  -- ensure y is positive
  (16 : ℝ) / y = 12 / 9 → 
  y = 12 := by
sorry

end similar_right_triangles_l1778_177811


namespace leftHandedWomenPercentage_l1778_177865

/-- Represents the population of Smithtown -/
structure Population where
  rightHanded : ℕ
  leftHanded : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for a valid Smithtown population -/
def isValidPopulation (p : Population) : Prop :=
  p.rightHanded = 3 * p.leftHanded ∧
  p.men = 3 * p.women / 2 ∧
  p.rightHanded + p.leftHanded = p.men + p.women

/-- A population with maximized right-handed men -/
def hasMaximizedRightHandedMen (p : Population) : Prop :=
  p.men = p.rightHanded

/-- Theorem: In a valid Smithtown population with maximized right-handed men,
    left-handed women constitute 25% of the total population -/
theorem leftHandedWomenPercentage (p : Population) 
  (hValid : isValidPopulation p) 
  (hMax : hasMaximizedRightHandedMen p) : 
  (p.leftHanded : ℚ) / (p.rightHanded + p.leftHanded : ℚ) = 1/4 := by
  sorry

end leftHandedWomenPercentage_l1778_177865


namespace n_good_lower_bound_two_is_seven_good_l1778_177869

/-- A tournament between n players where each player plays against every other player once --/
structure Tournament (n : ℕ) where
  result : Fin n → Fin n → Bool
  irreflexive : ∀ i, result i i = false
  antisymmetric : ∀ i j, result i j = !result j i

/-- A number k is n-good if there exists a tournament where for any k players, 
    there is another player who has lost to all of them --/
def is_n_good (n k : ℕ) : Prop :=
  ∃ t : Tournament n, ∀ (s : Finset (Fin n)) (hs : s.card = k),
    ∃ p : Fin n, p ∉ s ∧ ∀ q ∈ s, t.result q p = true

/-- The main theorem: For any n-good number k, n ≥ 2^(k+1) - 1 --/
theorem n_good_lower_bound (n k : ℕ) (h : is_n_good n k) : n ≥ 2^(k+1) - 1 :=
  sorry

/-- The smallest n for which 2 is n-good is 7 --/
theorem two_is_seven_good : 
  (is_n_good 7 2) ∧ (∀ m < 7, ¬ is_n_good m 2) :=
  sorry

end n_good_lower_bound_two_is_seven_good_l1778_177869


namespace half_area_to_longest_side_l1778_177830

/-- Represents a parallelogram field with given dimensions and angles -/
structure ParallelogramField where
  side1 : Real
  side2 : Real
  angle1 : Real
  angle2 : Real

/-- Calculates the fraction of the area closer to the longest side of the parallelogram field -/
def fraction_to_longest_side (field : ParallelogramField) : Real :=
  sorry

/-- Theorem stating that for a parallelogram field with specific dimensions,
    the fraction of the area closer to the longest side is 1/2 -/
theorem half_area_to_longest_side :
  let field : ParallelogramField := {
    side1 := 120,
    side2 := 80,
    angle1 := π / 3,  -- 60 degrees in radians
    angle2 := 2 * π / 3  -- 120 degrees in radians
  }
  fraction_to_longest_side field = 1 / 2 := by
  sorry

end half_area_to_longest_side_l1778_177830


namespace binomial_coefficient_sum_ratio_bound_l1778_177835

theorem binomial_coefficient_sum_ratio_bound (n : ℕ+) :
  let a := 2^(n : ℕ)
  let b := 4^(n : ℕ)
  (b / a) + (a / b) ≥ (5 : ℝ) / 2 := by
sorry

end binomial_coefficient_sum_ratio_bound_l1778_177835


namespace max_cookies_buyable_l1778_177822

theorem max_cookies_buyable (total_money : ℚ) (pack_price : ℚ) (cookies_per_pack : ℕ) : 
  total_money = 20.75 ∧ pack_price = 1.75 ∧ cookies_per_pack = 2 →
  ⌊total_money / pack_price⌋ * cookies_per_pack = 22 := by
sorry

end max_cookies_buyable_l1778_177822


namespace investment_growth_l1778_177815

/-- The present value of an investment -/
def present_value : ℝ := 217474.41

/-- The future value of the investment -/
def future_value : ℝ := 600000

/-- The annual interest rate -/
def interest_rate : ℝ := 0.07

/-- The number of years for the investment -/
def years : ℕ := 15

/-- Theorem stating that the present value invested at the given interest rate
    for the specified number of years will result in the future value -/
theorem investment_growth (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |future_value - present_value * (1 + interest_rate) ^ years| < ε :=
sorry

end investment_growth_l1778_177815


namespace linear_equation_and_expression_l1778_177867

theorem linear_equation_and_expression (a : ℝ) : 
  (∀ x, (a - 1) * x^(|a|) - 3 = 0 → (a - 1) * x - 3 = 0) ∧ (a - 1 ≠ 0) →
  a = -1 ∧ -4 * a^2 - 2 * (a - (2 * a^2 - a + 2)) = 8 := by
sorry

end linear_equation_and_expression_l1778_177867


namespace machine_purchase_price_l1778_177850

/-- Represents the purchase price of the machine in rupees -/
def purchase_price : ℕ := sorry

/-- Represents the repair cost in rupees -/
def repair_cost : ℕ := 5000

/-- Represents the transportation charges in rupees -/
def transportation_charges : ℕ := 1000

/-- Represents the profit percentage -/
def profit_percentage : ℚ := 50 / 100

/-- Represents the selling price in rupees -/
def selling_price : ℕ := 27000

/-- Theorem stating that the purchase price is 12000 rupees -/
theorem machine_purchase_price : 
  purchase_price = 12000 ∧
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price :=
sorry

end machine_purchase_price_l1778_177850


namespace q_satisfies_conditions_l1778_177842

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℝ) : ℝ := -x^2 - 6*x + 27

/-- Theorem stating that q satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-9) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end q_satisfies_conditions_l1778_177842


namespace roof_area_l1778_177870

theorem roof_area (width length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 4 * width → 
  length - width = 39 → 
  width * length = 676 := by
sorry

end roof_area_l1778_177870


namespace pepperoni_coverage_l1778_177804

/-- Represents a circular pizza with pepperoni toppings -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_count : ℕ
  pepperoni_across_diameter : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def fraction_covered (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni -/
theorem pepperoni_coverage (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across_diameter = 9)
  (h3 : p.pepperoni_count = 40) : 
  fraction_covered p = 40 / 81 := by
  sorry

end pepperoni_coverage_l1778_177804


namespace triangle_special_sequence_l1778_177802

theorem triangle_special_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  ∃ (α d : ℝ), A = α ∧ B = α + d ∧ C = α + 2*d ∧
  -- Sum of angles is π
  A + B + C = π ∧
  -- Reciprocals of sides form an arithmetic sequence
  2 * (1/b) = 1/a + 1/c ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Conclusion: all angles are π/3
  A = π/3 ∧ B = π/3 ∧ C = π/3 := by
sorry

end triangle_special_sequence_l1778_177802


namespace firing_time_per_minute_l1778_177839

/-- Calculates the time spent firing per minute given the firing interval and duration -/
def timeSpentFiring (secondsPerMinute : ℕ) (firingInterval : ℕ) (fireDuration : ℕ) : ℕ :=
  (secondsPerMinute / firingInterval) * fireDuration

/-- Proves that given the specified conditions, the time spent firing per minute is 20 seconds -/
theorem firing_time_per_minute :
  timeSpentFiring 60 15 5 = 20 := by sorry

end firing_time_per_minute_l1778_177839


namespace system_solution_l1778_177807

theorem system_solution :
  ∃ (x y : ℝ), (1/2 * x - 3/2 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end system_solution_l1778_177807


namespace intersection_implies_a_equals_one_l1778_177837

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  A ∩ B = {3} → a = 1 := by
sorry

end intersection_implies_a_equals_one_l1778_177837


namespace number_problem_l1778_177856

theorem number_problem (x : ℝ) : 5 * x + 4 = 19 → x = 3 := by
  sorry

end number_problem_l1778_177856


namespace fair_hair_percentage_l1778_177896

/-- Proves that if 32% of employees are women with fair hair and 40% of fair-haired employees are women, then 80% of employees have fair hair. -/
theorem fair_hair_percentage (total_employees : ℝ) (women_fair_hair : ℝ) (fair_hair : ℝ)
  (h1 : women_fair_hair = 0.32 * total_employees)
  (h2 : women_fair_hair = 0.40 * fair_hair) :
  fair_hair / total_employees = 0.80 := by
sorry

end fair_hair_percentage_l1778_177896


namespace fraction_difference_equals_nine_twentieths_l1778_177857

theorem fraction_difference_equals_nine_twentieths :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end fraction_difference_equals_nine_twentieths_l1778_177857


namespace parabola_directrix_l1778_177876

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d ∧ 
  y_d = -5/4 :=
sorry

end parabola_directrix_l1778_177876


namespace william_final_napkins_l1778_177828

def initial_napkins : ℕ := 15
def olivia_napkins : ℕ := 10
def amelia_napkins : ℕ := 2 * olivia_napkins

theorem william_final_napkins :
  initial_napkins + olivia_napkins + amelia_napkins = 45 :=
by sorry

end william_final_napkins_l1778_177828


namespace floor_sqrt_equality_l1778_177847

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧
  ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by sorry

end floor_sqrt_equality_l1778_177847


namespace log_meaningful_range_l1778_177899

/-- The range of real number a for which log_(a-1)(5-a) is meaningful -/
def meaningful_log_range : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 5}

theorem log_meaningful_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) ^ x = 5 - a) ↔ a ∈ meaningful_log_range := by
  sorry

end log_meaningful_range_l1778_177899


namespace only_elevator_is_pure_translation_l1778_177809

/-- Represents a physical phenomenon --/
inductive Phenomenon
  | RollingSoccerBall
  | RotatingFanBlades
  | ElevatorGoingUp
  | MovingCarRearWheel

/-- Defines whether a phenomenon exhibits pure translation --/
def isPureTranslation (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.ElevatorGoingUp => True
  | _ => False

/-- The rolling soccer ball involves both rotation and translation --/
axiom rolling_soccer_ball_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RollingSoccerBall

/-- Rotating fan blades involve rotation around a central axis --/
axiom rotating_fan_blades_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RotatingFanBlades

/-- An elevator going up moves from one level to another without rotating --/
axiom elevator_going_up_is_pure_translation :
  isPureTranslation Phenomenon.ElevatorGoingUp

/-- A moving car rear wheel primarily exhibits rotation --/
axiom moving_car_rear_wheel_not_pure_translation :
  ¬ isPureTranslation Phenomenon.MovingCarRearWheel

/-- Theorem: Only the elevator going up exhibits pure translation --/
theorem only_elevator_is_pure_translation :
  ∀ p : Phenomenon, isPureTranslation p ↔ p = Phenomenon.ElevatorGoingUp :=
by sorry


end only_elevator_is_pure_translation_l1778_177809


namespace remainder_problem_l1778_177824

theorem remainder_problem (x : ℤ) : x % 9 = 2 → x % 63 = 7 := by
  sorry

end remainder_problem_l1778_177824


namespace minimum_training_months_l1778_177821

/-- The distance of a marathon in miles -/
def marathonDistance : ℝ := 26.3

/-- The initial running distance in miles -/
def initialDistance : ℝ := 3

/-- The function that calculates the running distance after a given number of months -/
def runningDistance (months : ℕ) : ℝ :=
  initialDistance * (2 ^ months)

/-- The theorem stating that 5 months is the minimum number of months needed to run a marathon -/
theorem minimum_training_months :
  (∀ m : ℕ, m < 5 → runningDistance m < marathonDistance) ∧
  (runningDistance 5 ≥ marathonDistance) := by
  sorry

#check minimum_training_months

end minimum_training_months_l1778_177821


namespace range_of_f_l1778_177881

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem range_of_f :
  ∃ (a b : ℝ), a = -2 ∧ b = 2 ∧
  (∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = y) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end range_of_f_l1778_177881


namespace second_quadrant_m_negative_l1778_177801

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The theorem stating that if a point P(m, 2) is in the second quadrant, then m < 0 -/
theorem second_quadrant_m_negative (m : ℝ) :
  SecondQuadrant ⟨m, 2⟩ → m < 0 := by
  sorry

end second_quadrant_m_negative_l1778_177801


namespace fly_path_distance_l1778_177836

theorem fly_path_distance (radius : ℝ) (third_segment : ℝ) :
  radius = 60 ∧ third_segment = 90 →
  ∃ (second_segment : ℝ),
    second_segment^2 + third_segment^2 = (2 * radius)^2 ∧
    (2 * radius) + third_segment + second_segment = 120 + 90 + 30 * Real.sqrt 7 :=
by sorry

end fly_path_distance_l1778_177836


namespace quadratic_function_value_l1778_177838

-- Define the quadratic function f(x) = x² - ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - a*x + b

-- State the theorem
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = -1 → f a b 2 = 2 → f a b (-4) = 14 := by
  sorry

end quadratic_function_value_l1778_177838


namespace last_score_is_84_l1778_177827

def scores : List ℕ := [68, 75, 78, 84, 85, 90]

def is_valid_last_score (s : ℕ) : Prop :=
  s ∈ scores ∧
  ∀ subset : List ℕ, subset.length < 6 → subset ⊆ scores →
  (subset.sum + s) % (subset.length + 1) = 0

theorem last_score_is_84 :
  ∀ s ∈ scores, is_valid_last_score s ↔ s = 84 := by sorry

end last_score_is_84_l1778_177827


namespace seven_power_plus_one_prime_factors_l1778_177894

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    (primes.card ≥ 2 * n + 3) ∧ 
    ((primes.prod id) = 7^(7^n) + 1) :=
sorry

end seven_power_plus_one_prime_factors_l1778_177894


namespace parallel_lines_m_value_l1778_177863

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 * b2 = a2 * b1) ∧ (a1 ≠ 0 ∨ a2 ≠ 0)

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  parallel_lines 1 (2*m) (-1) (m-2) (-m) 2 → m = 3/2 := by
  sorry

end parallel_lines_m_value_l1778_177863


namespace coordinate_and_vector_problem_l1778_177810

-- Define the points and vectors
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-2, 1)  -- Calculated from |OB| = √5 and x = -2
def O : ℝ × ℝ := (0, 0)

-- Define the rotation function
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the vector OP
def OP : ℝ × ℝ := (2, 6)  -- Calculated from |OP| = 2√10 and cos θ = √10/10

-- Define the theorem
theorem coordinate_and_vector_problem :
  let C := rotate90Clockwise (B.1 - O.1, B.2 - O.2)
  let x := ((OP.1 * B.2) - (OP.2 * B.1)) / ((A.1 * B.2) - (A.2 * B.1))
  let y := ((OP.1 * A.2) - (OP.2 * A.1)) / ((B.1 * A.2) - (B.2 * A.1))
  C = (1, 2) ∧ x + y = 8 := by
  sorry

end coordinate_and_vector_problem_l1778_177810


namespace sum_of_two_equals_third_l1778_177820

theorem sum_of_two_equals_third (a b c x y : ℝ) 
  (h1 : (a + x)⁻¹ = 6)
  (h2 : (b + y)⁻¹ = 3)
  (h3 : (c + x + y)⁻¹ = 2) : 
  c = a + b := by sorry

end sum_of_two_equals_third_l1778_177820


namespace no_primes_in_factorial_range_l1778_177886

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end no_primes_in_factorial_range_l1778_177886


namespace no_real_solutions_l1778_177877

theorem no_real_solutions :
  ¬∃ (z : ℝ), (3*z - 9*z + 27)^2 + 4 = -2*(abs z) := by
  sorry

end no_real_solutions_l1778_177877


namespace units_digit_of_p_l1778_177814

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  (0 < units_digit p) → 
  (units_digit (p^3) - units_digit (p^2) = 0) →
  (units_digit (p + 5) = 1) →
  units_digit p = 6 :=
by sorry

end units_digit_of_p_l1778_177814


namespace train_tunnel_time_l1778_177861

/-- The time taken for a train to pass through a tunnel -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length = 1.1 →
  (train_length + tunnel_length * 1000) / (train_speed_kmh * 1000 / 3600) / 60 = 1 := by
  sorry

end train_tunnel_time_l1778_177861


namespace buffalo_count_is_two_l1778_177845

/-- Represents the number of animals seen on each day of Erica's safari --/
structure SafariCount where
  saturday : ℕ
  sunday_leopards : ℕ
  sunday_buffaloes : ℕ
  monday : ℕ

/-- The total number of animals seen during the safari --/
def total_animals : ℕ := 20

/-- The actual count of animals seen on each day --/
def safari_count : SafariCount where
  saturday := 5  -- 3 lions + 2 elephants
  sunday_leopards := 5
  sunday_buffaloes := 2  -- This is what we want to prove
  monday := 8  -- 5 rhinos + 3 warthogs

theorem buffalo_count_is_two :
  safari_count.sunday_buffaloes = 2 :=
by
  sorry

#check buffalo_count_is_two

end buffalo_count_is_two_l1778_177845


namespace unique_divisible_number_l1778_177874

def number (d : Nat) : Nat := 62684400 + d * 10

theorem unique_divisible_number :
  ∃! d : Nat, d < 10 ∧ (number d).mod 8 = 0 ∧ (number d).mod 5 = 0 :=
sorry

end unique_divisible_number_l1778_177874


namespace f_expression_l1778_177813

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, x ≥ 0 → f (Real.sqrt x + 1) = x + 3) →
  (∀ x : ℝ, x ≥ 0 → f (x + 1) = x^2 + 3) :=
by sorry

end f_expression_l1778_177813


namespace food_distribution_l1778_177853

theorem food_distribution (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℝ) (remaining_days : ℕ) :
  initial_men = 760 →
  initial_days = 22 →
  additional_men = 134.11764705882354 →
  remaining_days = 17 →
  ∃ (x : ℝ),
    x = 2 ∧
    (initial_men : ℝ) * (initial_days : ℝ) = 
      (initial_men : ℝ) * x + ((initial_men : ℝ) + additional_men) * (remaining_days : ℝ) :=
by sorry

end food_distribution_l1778_177853


namespace number_difference_l1778_177866

theorem number_difference (a b : ℕ) : 
  a + b = 25800 →
  ∃ k : ℕ, b = 12 * k →
  a = k →
  b - a = 21824 :=
by
  sorry

end number_difference_l1778_177866


namespace dispatch_plans_count_l1778_177833

theorem dispatch_plans_count : ∀ (n m k : ℕ),
  n = 6 → m = 4 → k = 2 →
  (Nat.choose n k) * (n - k) * (n - k - 1) = 180 :=
by sorry

end dispatch_plans_count_l1778_177833


namespace carrots_thrown_out_l1778_177882

theorem carrots_thrown_out (initial_carrots : ℕ) (additional_carrots : ℕ) (remaining_carrots : ℕ) : 
  initial_carrots = 48 →
  additional_carrots = 15 →
  remaining_carrots = 52 →
  initial_carrots + additional_carrots - remaining_carrots = 11 := by
sorry

end carrots_thrown_out_l1778_177882


namespace solve_equation_l1778_177891

theorem solve_equation : ∃ x : ℝ, 3 * x - 6 = |(-23 + 5)|^2 ∧ x = 110 := by
  sorry

end solve_equation_l1778_177891


namespace non_negative_integer_solutions_count_solution_count_equals_10626_l1778_177841

theorem non_negative_integer_solutions_count : Nat :=
  let n : Nat := 20
  let k : Nat := 5
  (n + k - 1).choose (k - 1)

theorem solution_count_equals_10626 : non_negative_integer_solutions_count = 10626 := by
  sorry

end non_negative_integer_solutions_count_solution_count_equals_10626_l1778_177841


namespace apple_cost_18_pounds_l1778_177848

/-- The cost of apples given a rate and a quantity -/
def apple_cost (rate_dollars : ℚ) (rate_pounds : ℚ) (quantity : ℚ) : ℚ :=
  (rate_dollars / rate_pounds) * quantity

/-- Theorem: The cost of 18 pounds of apples at a rate of 5 dollars per 6 pounds is 15 dollars -/
theorem apple_cost_18_pounds : apple_cost 5 6 18 = 15 := by
  sorry

end apple_cost_18_pounds_l1778_177848


namespace base_10_500_equals_base_6_2152_l1778_177885

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_500_equals_base_6_2152 :
  toBase6 500 = [2, 1, 5, 2] ∧ fromBase6 [2, 1, 5, 2] = 500 :=
sorry

end base_10_500_equals_base_6_2152_l1778_177885


namespace overall_profit_calculation_l1778_177829

/-- Calculates the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ)
  (h1 : refrigerator_cost = 15000)
  (h2 : mobile_cost = 8000)
  (h3 : refrigerator_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100) :
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost)).floor = 200 :=
by sorry

end overall_profit_calculation_l1778_177829
