import Mathlib

namespace triangle_angle_with_sine_half_l3215_321518

theorem triangle_angle_with_sine_half (α : Real) :
  0 < α ∧ α < π ∧ Real.sin α = 1/2 → α = π/6 ∨ α = 5*π/6 := by sorry

end triangle_angle_with_sine_half_l3215_321518


namespace heart_properties_l3215_321567

def heart (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem heart_properties :
  (∀ x y : ℝ, heart x y = heart y x) ∧
  (∃ x y : ℝ, 2 * (heart x y) ≠ heart (2*x) (2*y)) ∧
  (∀ x : ℝ, heart x 0 = x^2) ∧
  (∀ x : ℝ, heart x x = 0) ∧
  (∀ x y : ℝ, x ≠ y → heart x y > 0) :=
by sorry

end heart_properties_l3215_321567


namespace value_calculation_l3215_321558

theorem value_calculation (number : ℕ) (value : ℕ) (h1 : number = 16) (h2 : value = 2 * number - 12) : value = 20 := by
  sorry

end value_calculation_l3215_321558


namespace definite_integral_exp_plus_2x_l3215_321513

theorem definite_integral_exp_plus_2x : ∫ (x : ℝ) in (0)..(1), (Real.exp x + 2 * x) = Real.exp 1 := by
  sorry

end definite_integral_exp_plus_2x_l3215_321513


namespace sequence_limit_uniqueness_l3215_321543

theorem sequence_limit_uniqueness (a : ℕ → ℝ) (l₁ l₂ : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₁| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₂| < ε) →
  l₁ = l₂ :=
by sorry

end sequence_limit_uniqueness_l3215_321543


namespace inscribed_hexagon_area_l3215_321598

theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 576 * Real.pi) :
  let r := Real.sqrt (circle_area / Real.pi)
  let hexagon_area := 6 * ((r^2 * Real.sqrt 3) / 4)
  hexagon_area = 864 * Real.sqrt 3 := by sorry

end inscribed_hexagon_area_l3215_321598


namespace binomial_60_3_l3215_321541

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end binomial_60_3_l3215_321541


namespace caz_at_position_p_l3215_321504

-- Define the type for positions in the gallery
inductive Position
| P
| Other

-- Define the type for people in the gallery
inductive Person
| Ali
| Bea
| Caz
| Dan

-- Define the visibility relation
def CanSee (a b : Person) : Prop := sorry

-- Define the position of a person
def IsAt (p : Person) (pos : Position) : Prop := sorry

-- State the theorem
theorem caz_at_position_p :
  -- Conditions
  (∀ x, x ≠ Person.Ali → ¬CanSee Person.Ali x) →
  (CanSee Person.Bea Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Bea x) →
  (CanSee Person.Caz Person.Bea) →
  (CanSee Person.Caz Person.Dan) →
  (∀ x, x ≠ Person.Bea ∧ x ≠ Person.Dan → ¬CanSee Person.Caz x) →
  (CanSee Person.Dan Person.Caz) →
  (∀ x, x ≠ Person.Caz → ¬CanSee Person.Dan x) →
  -- Conclusion
  IsAt Person.Caz Position.P :=
by sorry

end caz_at_position_p_l3215_321504


namespace partial_fraction_decomposition_l3215_321532

theorem partial_fraction_decomposition (x A B C : ℝ) :
  x ≠ 2 → x ≠ 4 →
  (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) ↔
  (A = 20 ∧ B = -15 ∧ C = -10) :=
by sorry

end partial_fraction_decomposition_l3215_321532


namespace hyperbola_equation_l3215_321525

/-- The standard equation of a hyperbola with given focus and conjugate axis endpoint -/
theorem hyperbola_equation (f : ℝ × ℝ) (e : ℝ × ℝ) :
  f = (-10, 0) →
  e = (0, 4) →
  ∀ x y : ℝ, (x^2 / 84 - y^2 / 16 = 1) ↔ 
    (∃ a b c : ℝ, a^2 = 84 ∧ b^2 = 16 ∧ c^2 = a^2 + b^2 ∧
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      c = 10 ∧ 
      (x - f.1)^2 + (y - f.2)^2 - ((x + 10)^2 + y^2) = 4 * a^2) :=
by sorry

end hyperbola_equation_l3215_321525


namespace andrew_fruit_purchase_cost_l3215_321570

/-- Calculates the total cost of fruits purchased by Andrew -/
theorem andrew_fruit_purchase_cost : 
  let grapes_quantity : ℕ := 14
  let grapes_price : ℕ := 54
  let mangoes_quantity : ℕ := 10
  let mangoes_price : ℕ := 62
  let pineapple_quantity : ℕ := 8
  let pineapple_price : ℕ := 40
  let kiwi_quantity : ℕ := 5
  let kiwi_price : ℕ := 30
  let total_cost := 
    grapes_quantity * grapes_price + 
    mangoes_quantity * mangoes_price + 
    pineapple_quantity * pineapple_price + 
    kiwi_quantity * kiwi_price
  total_cost = 1846 := by
  sorry


end andrew_fruit_purchase_cost_l3215_321570


namespace square_area_is_400_l3215_321531

-- Define the radius of the circles
def circle_radius : ℝ := 5

-- Define the side length of the square
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- Theorem: The area of the square is 400 square inches
theorem square_area_is_400 : square_side_length ^ 2 = 400 := by
  sorry


end square_area_is_400_l3215_321531


namespace marked_price_is_correct_l3215_321599

/-- The marked price of a down jacket -/
def marked_price : ℝ := 550

/-- The cost price of the down jacket -/
def cost_price : ℝ := 350

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit made on the sale -/
def profit : ℝ := 90

/-- Theorem stating that the marked price is correct given the conditions -/
theorem marked_price_is_correct : 
  selling_percentage * marked_price - cost_price = profit :=
by sorry

end marked_price_is_correct_l3215_321599


namespace sum_of_squares_l3215_321507

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + c * a = 6) (h2 : a + b + c = 15) :
  a^2 + b^2 + c^2 = 213 := by
  sorry

end sum_of_squares_l3215_321507


namespace largest_divisor_of_factorial_l3215_321561

theorem largest_divisor_of_factorial (m n : ℕ) (hm : m ≥ 3) (hn : n > m * (m - 2)) :
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d)) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ d) ∧
    ∀ d' > 0, d' ∣ n.factorial → (∀ k ∈ Finset.Icc m n, ¬(k ∣ d')) → d' ≤ d) →
  (m - 1 : ℕ) > 0 ∧ (m - 1 : ℕ) ∣ n.factorial ∧ ∀ k ∈ Finset.Icc m n, ¬(k ∣ (m - 1 : ℕ)) :=
by sorry

end largest_divisor_of_factorial_l3215_321561


namespace smallest_n_years_for_90_percent_depreciation_l3215_321536

-- Define the depreciation rate
def depreciation_rate : ℝ := 0.9

-- Define the target depreciation
def target_depreciation : ℝ := 0.1

-- Define the approximation of log 3
def log3_approx : ℝ := 0.477

-- Define the function to check if n years of depreciation meets the target
def meets_target (n : ℕ) : Prop := depreciation_rate ^ n ≤ target_depreciation

-- Statement to prove
theorem smallest_n_years_for_90_percent_depreciation :
  ∃ n : ℕ, meets_target n ∧ ∀ m : ℕ, m < n → ¬meets_target m :=
sorry

end smallest_n_years_for_90_percent_depreciation_l3215_321536


namespace courtyard_length_l3215_321593

/-- The length of a rectangular courtyard given its width and paving stones --/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ)
  (h_width : width = 16.5)
  (h_num_stones : num_stones = 165)
  (h_stone_length : stone_length = 2.5)
  (h_stone_width : stone_width = 2) :
  width * (num_stones * stone_length * stone_width / width) = 50 := by
  sorry

#check courtyard_length

end courtyard_length_l3215_321593


namespace income_percentage_difference_l3215_321500

/-- Given the monthly incomes of A and B in ratio 5:2, C's monthly income of 15000,
    and A's annual income of 504000, prove that B's monthly income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_monthly B_monthly C_monthly : ℕ),
    C_monthly = 15000 →
    A_monthly * 12 = 504000 →
    A_monthly * 2 = B_monthly * 5 →
    (B_monthly - C_monthly) * 100 = C_monthly * 12 := by
  sorry

end income_percentage_difference_l3215_321500


namespace opposite_of_A_is_F_l3215_321560

-- Define the labels for the cube faces
inductive CubeFace
  | A | B | C | D | E | F

-- Define a structure for the cube
structure Cube where
  faces : Finset CubeFace
  opposite : CubeFace → CubeFace

-- Define the properties of the cube
axiom cube_has_six_faces : ∀ (c : Cube), c.faces.card = 6

axiom cube_has_unique_opposite : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ∈ c.faces ∧ c.opposite (c.opposite f) = f

axiom cube_opposite_distinct : ∀ (c : Cube) (f : CubeFace), 
  f ∈ c.faces → c.opposite f ≠ f

-- Theorem to prove
theorem opposite_of_A_is_F (c : Cube) : 
  CubeFace.A ∈ c.faces → c.opposite CubeFace.A = CubeFace.F := by
  sorry

end opposite_of_A_is_F_l3215_321560


namespace sum_of_absolute_coefficients_l3215_321529

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end sum_of_absolute_coefficients_l3215_321529


namespace solution_set_abs_x_times_x_minus_two_l3215_321542

theorem solution_set_abs_x_times_x_minus_two (x : ℝ) :
  {x : ℝ | |x| * (x - 2) ≥ 0} = {x : ℝ | x ≥ 2 ∨ x = 0} :=
by sorry

end solution_set_abs_x_times_x_minus_two_l3215_321542


namespace shaded_area_theorem_l3215_321526

theorem shaded_area_theorem (total_area : ℝ) (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_area = 64 → 
  total_triangles = 64 → 
  shaded_triangles = 28 → 
  (shaded_triangles : ℝ) * (total_area / total_triangles) = 28 := by
  sorry

end shaded_area_theorem_l3215_321526


namespace gravitational_force_on_space_station_l3215_321557

/-- Gravitational force model -/
structure GravitationalModel where
  k : ℝ
  force : ℝ → ℝ
  h_inverse_square : ∀ d, d > 0 → force d = k / (d^2)

/-- Problem statement -/
theorem gravitational_force_on_space_station
  (model : GravitationalModel)
  (h_surface : model.force 6000 = 800)
  : model.force 360000 = 2/9 := by
  sorry


end gravitational_force_on_space_station_l3215_321557


namespace range_of_m_l3215_321509

-- Define set A
def A : Set ℝ := {y | ∃ x > 0, y = 1 / x}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 * x - 4)}

-- Theorem statement
theorem range_of_m (m : ℝ) (h1 : m ∈ A) (h2 : m ∉ B) : m ∈ Set.Ioo 0 2 := by
  sorry

end range_of_m_l3215_321509


namespace pyramid_lateral_surface_area_l3215_321528

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- Calculate the lateral surface area of a regular square pyramid -/
def lateral_surface_area (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a regular square pyramid with 
    base edge length 2√2 cm and volume 8 cm³ is 4√22 cm² -/
theorem pyramid_lateral_surface_area :
  let p : RegularSquarePyramid := ⟨2 * Real.sqrt 2, 8⟩
  lateral_surface_area p = 4 * Real.sqrt 22 := by
  sorry

end pyramid_lateral_surface_area_l3215_321528


namespace cost_difference_l3215_321522

def ice_cream_cartons : ℕ := 100
def yoghurt_cartons : ℕ := 35
def ice_cream_cost_per_carton : ℚ := 12
def yoghurt_cost_per_carton : ℚ := 3
def ice_cream_discount_rate : ℚ := 0.05
def yoghurt_tax_rate : ℚ := 0.08

def ice_cream_total_cost : ℚ := ice_cream_cartons * ice_cream_cost_per_carton
def yoghurt_total_cost : ℚ := yoghurt_cartons * yoghurt_cost_per_carton

def ice_cream_discounted_cost : ℚ := ice_cream_total_cost * (1 - ice_cream_discount_rate)
def yoghurt_taxed_cost : ℚ := yoghurt_total_cost * (1 + yoghurt_tax_rate)

theorem cost_difference : 
  ice_cream_discounted_cost - yoghurt_taxed_cost = 1026.60 := by
  sorry

end cost_difference_l3215_321522


namespace area_range_of_special_triangle_l3215_321591

/-- Given an acute triangle ABC where angles A, B, C form an arithmetic sequence
    and the side opposite to angle B has length √3, prove that the area S of the triangle
    satisfies √3/2 < S ≤ 3√3/4. -/
theorem area_range_of_special_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧  -- sum of angles in a triangle
  2 * B = A + C ∧  -- A, B, C form an arithmetic sequence
  b = Real.sqrt 3 ∧  -- side opposite to B has length √3
  S = (1 / 2) * a * c * Real.sin B ∧  -- area formula
  a * Real.sin B = b * Real.sin A ∧  -- sine law
  c * Real.sin B = b * Real.sin C  -- sine law
  →
  Real.sqrt 3 / 2 < S ∧ S ≤ 3 * Real.sqrt 3 / 4 := by
  sorry


end area_range_of_special_triangle_l3215_321591


namespace train_length_l3215_321590

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 170 := by
  sorry

end train_length_l3215_321590


namespace largest_prime_factor_of_S_l3215_321566

/-- The product of all non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) for n from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (fun i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- 103 is the largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (m : ℕ), S = 103 * m ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ S → q ≤ 103 :=
  sorry

end largest_prime_factor_of_S_l3215_321566


namespace expression_equals_negative_one_l3215_321545

theorem expression_equals_negative_one 
  (x y z : ℝ) 
  (hx : x ≠ 1) 
  (hy : y ≠ 2) 
  (hz : z ≠ 3) : 
  (x - 1) / (3 - z) * (y - 2) / (1 - x) * (z - 3) / (2 - y) = -1 := by
  sorry

end expression_equals_negative_one_l3215_321545


namespace weekend_price_is_105_l3215_321568

def original_price : ℝ := 250
def sale_discount : ℝ := 0.4
def weekend_discount : ℝ := 0.3

def sale_price : ℝ := original_price * (1 - sale_discount)
def weekend_price : ℝ := sale_price * (1 - weekend_discount)

theorem weekend_price_is_105 : weekend_price = 105 := by sorry

end weekend_price_is_105_l3215_321568


namespace tangent_slope_angle_at_zero_l3215_321585

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * cos x

theorem tangent_slope_angle_at_zero (α : ℝ) :
  (∀ x, HasDerivAt f (exp x * (cos x - sin x)) x) →
  HasDerivAt f 1 0 →
  0 ≤ α →
  α < π →
  tan α = 1 →
  α = π / 4 :=
by sorry

end tangent_slope_angle_at_zero_l3215_321585


namespace always_greater_than_m_l3215_321544

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end always_greater_than_m_l3215_321544


namespace interest_rate_problem_l3215_321553

/-- Given a principal amount and an interest rate, if increasing the interest rate by 3%
    results in 210 more interest over 10 years, then the principal amount must be 700. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 3) * 10 / 100 = P * R * 10 / 100 + 210) :
  P = 700 := by
  sorry

end interest_rate_problem_l3215_321553


namespace cubic_sum_product_l3215_321511

theorem cubic_sum_product (a b c : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a^2 + b^2 + c^2 = 15)
  (h3 : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end cubic_sum_product_l3215_321511


namespace octagon_perimeter_l3215_321527

/-- An octagon is a polygon with 8 sides -/
def Octagon : Type := Unit

/-- The length of each side of the octagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of its sides -/
def perimeter (p : Octagon) : ℝ := 8 * side_length

theorem octagon_perimeter : 
  ∀ (o : Octagon), perimeter o = 24 := by
  sorry

end octagon_perimeter_l3215_321527


namespace womens_doubles_handshakes_l3215_321556

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (num_teams : ℕ) (team_size : ℕ) : 
  num_teams = 4 → team_size = 2 → num_teams * team_size * (num_teams * team_size - team_size) / 2 = 24 := by
  sorry

end womens_doubles_handshakes_l3215_321556


namespace sin_cos_square_identity_l3215_321546

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end sin_cos_square_identity_l3215_321546


namespace initial_men_count_l3215_321575

/-- Proves that the initial number of men is 760, given the food supply conditions. -/
theorem initial_men_count (M : ℕ) : 
  (M * 22 = (M + 40) * 19 + M * 2) → M = 760 := by
  sorry

end initial_men_count_l3215_321575


namespace students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l3215_321554

/-- Represents the number of students interested in a combination of subjects -/
structure InterestCounts where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  physics : ℕ
  none : ℕ
  onlyBiology : ℕ
  onlyPhysics : ℕ
  biologyAndChemistry : ℕ

/-- The theorem stating the number of students interested in all three subjects -/
theorem students_interested_in_all_subjects (counts : InterestCounts) : ℕ :=
  let all_three := counts.biology + counts.chemistry + counts.physics -
    (counts.onlyBiology + counts.biologyAndChemistry + counts.onlyPhysics) - 
    (counts.total - counts.none)
  2

/-- The main theorem proving the number of students interested in all subjects -/
theorem prove_students_interested_in_all_subjects : 
  ∃ (counts : InterestCounts), 
    counts.total = 40 ∧ 
    counts.biology = 20 ∧ 
    counts.chemistry = 10 ∧ 
    counts.physics = 8 ∧ 
    counts.none = 7 ∧ 
    counts.onlyBiology = 12 ∧ 
    counts.onlyPhysics = 4 ∧ 
    counts.biologyAndChemistry = 6 ∧ 
    students_interested_in_all_subjects counts = 2 :=
by
  sorry

end students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l3215_321554


namespace incorrect_inequality_transformation_l3215_321586

theorem incorrect_inequality_transformation (a b : ℝ) (h : a > b) :
  ¬(1 - a > 1 - b) := by
  sorry

end incorrect_inequality_transformation_l3215_321586


namespace angle_between_vectors_is_pi_over_3_l3215_321533

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors_is_pi_over_3 (a b : ℝ × ℝ) 
  (h1 : a • (a + b) = 5)
  (h2 : ‖a‖ = 2)
  (h3 : ‖b‖ = 1) : 
  angle_between_vectors a b = π / 3 := by sorry

end angle_between_vectors_is_pi_over_3_l3215_321533


namespace modified_rubiks_cube_cubie_count_l3215_321577

/-- Represents a modified Rubik's cube with 8 corner cubies removed --/
structure ModifiedRubiksCube where
  /-- The number of small cubies with 4 painted faces --/
  four_face_cubies : Nat
  /-- The number of small cubies with 1 painted face --/
  one_face_cubies : Nat
  /-- The number of small cubies with 0 painted faces --/
  zero_face_cubies : Nat

/-- Theorem stating the correct number of cubies for each type in a modified Rubik's cube --/
theorem modified_rubiks_cube_cubie_count :
  ∃ (cube : ModifiedRubiksCube),
    cube.four_face_cubies = 12 ∧
    cube.one_face_cubies = 6 ∧
    cube.zero_face_cubies = 1 :=
by sorry

end modified_rubiks_cube_cubie_count_l3215_321577


namespace ceiling_floor_product_range_l3215_321588

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 120 → -11 < y ∧ y < -10 := by
  sorry

end ceiling_floor_product_range_l3215_321588


namespace wrong_number_calculation_l3215_321571

def wrong_number (n : ℕ) (initial_avg : ℚ) (correct_num : ℕ) (correct_avg : ℚ) : ℚ :=
  n * initial_avg + correct_num - (n * correct_avg)

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_num : ℕ) :
  n = 10 →
  initial_avg = 5 →
  correct_num = 36 →
  correct_avg = 6 →
  wrong_number n initial_avg correct_num correct_avg = 26 := by
  sorry

end wrong_number_calculation_l3215_321571


namespace reciprocal_of_five_eighths_l3215_321516

theorem reciprocal_of_five_eighths :
  let x : ℚ := 5 / 8
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = 8 / 5 := by
sorry

end reciprocal_of_five_eighths_l3215_321516


namespace penalty_kicks_count_l3215_321589

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → (total_players - goalies) * goalies = 96 := by
  sorry

end penalty_kicks_count_l3215_321589


namespace sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l3215_321517

theorem sum_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℝ := -18
  let b : ℝ := 54
  let c : ℝ := -72
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l3215_321517


namespace wood_length_equation_l3215_321594

/-- Represents the length of a piece of wood that satisfies the measurement conditions. -/
def wood_length (x : ℝ) : Prop :=
  ∃ (rope_length : ℝ),
    rope_length - x = 4.5 ∧
    (rope_length / 2) - x = 1

/-- Proves that the wood length satisfies the equation from the problem. -/
theorem wood_length_equation (x : ℝ) :
  wood_length x → (x + 4.5) / 2 = x - 1 := by
  sorry

end wood_length_equation_l3215_321594


namespace digit_sum_problem_l3215_321582

theorem digit_sum_problem (P Q : ℕ) (h1 : P < 10) (h2 : Q < 10) 
  (h3 : 1013 + 1000 * P + 100 * Q + 10 * P + Q = 2023) : P + Q = 1 := by
  sorry

end digit_sum_problem_l3215_321582


namespace sachin_age_l3215_321565

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 12 = rahul_age * 5) :
  sachin_age = 5 := by
sorry

end sachin_age_l3215_321565


namespace sum_of_triangles_eq_22_l3215_321555

/-- Represents the value of a triangle with vertices a, b, and c -/
def triangle_value (a b c : ℕ) : ℕ := a * b + c

/-- The sum of the values of two specific triangles -/
def sum_of_triangles : ℕ :=
  triangle_value 3 2 5 + triangle_value 4 1 7

theorem sum_of_triangles_eq_22 : sum_of_triangles = 22 := by
  sorry

end sum_of_triangles_eq_22_l3215_321555


namespace max_substances_l3215_321578

/-- The number of substances generated when ethane is mixed with chlorine gas under lighting conditions -/
def num_substances : ℕ := sorry

/-- The number of isomers for monochloroethane -/
def mono_isomers : ℕ := 1

/-- The number of isomers for dichloroethane (including geometric isomers) -/
def di_isomers : ℕ := 3

/-- The number of isomers for trichloroethane -/
def tri_isomers : ℕ := 2

/-- The number of isomers for tetrachloroethane -/
def tetra_isomers : ℕ := 2

/-- The number of isomers for pentachloroethane -/
def penta_isomers : ℕ := 1

/-- The number of isomers for hexachloroethane -/
def hexa_isomers : ℕ := 1

/-- Hydrogen chloride is also formed -/
def hcl_formed : Prop := true

theorem max_substances :
  num_substances = mono_isomers + di_isomers + tri_isomers + tetra_isomers + penta_isomers + hexa_isomers + 1 ∧
  num_substances = 10 := by sorry

end max_substances_l3215_321578


namespace intersection_slope_l3215_321579

/-- Given two lines p and q that intersect at (1, 1), prove that the slope of q is -3 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = -2*x + 3 → y = k*x + 4) → -- Line p: y = -2x + 3, Line q: y = kx + 4
  1 = -2*1 + 3 →                            -- (1, 1) satisfies line p
  1 = k*1 + 4 →                             -- (1, 1) satisfies line q
  k = -3 :=
by sorry

end intersection_slope_l3215_321579


namespace trishul_investment_percentage_l3215_321547

/-- Proves that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  vishal + trishul + raghu = 6069 →  -- Total sum of investments
  raghu = 2100 →  -- Raghu's investment
  (raghu - trishul) / raghu = 0.1 :=  -- Trishul invested 10% less than Raghu
by sorry

end trishul_investment_percentage_l3215_321547


namespace tv_installment_plan_duration_l3215_321572

theorem tv_installment_plan_duration (cash_price down_payment monthly_payment cash_savings : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  cash_savings = 80 →
  (cash_price + cash_savings - down_payment) / monthly_payment = 12 :=
by
  sorry

end tv_installment_plan_duration_l3215_321572


namespace third_meeting_at_45km_l3215_321502

/-- Two people moving with constant speeds on a 100 km path between points A and B -/
structure TwoMovers :=
  (speed_ratio : ℚ)
  (first_meet : ℚ)
  (second_meet : ℚ)

/-- The third meeting point of two movers given their speed ratio and first two meeting points -/
def third_meeting_point (m : TwoMovers) : ℚ :=
  100 - (3 / 8) * 200

/-- Theorem stating that under given conditions, the third meeting point is 45 km from A -/
theorem third_meeting_at_45km (m : TwoMovers) 
  (h1 : m.first_meet = 20)
  (h2 : m.second_meet = 80)
  (h3 : m.speed_ratio = 3 / 5) :
  third_meeting_point m = 45 := by
  sorry

#eval third_meeting_point { speed_ratio := 3 / 5, first_meet := 20, second_meet := 80 }

end third_meeting_at_45km_l3215_321502


namespace number_equation_solution_l3215_321583

theorem number_equation_solution : 
  ∃ x : ℝ, (0.75 * x + 2 = 8) ∧ (x = 8) := by
  sorry

end number_equation_solution_l3215_321583


namespace smallest_divisible_number_after_2013_l3215_321535

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 ∧ i < 10 → n % i = 0

theorem smallest_divisible_number_after_2013 :
  ∃ (n : ℕ),
    n ≥ 2013000 ∧
    is_divisible_by_all_less_than_10 n ∧
    (∀ m : ℕ, 2013000 ≤ m ∧ m < n → ¬is_divisible_by_all_less_than_10 m) ∧
    n = 2013480 :=
sorry

end smallest_divisible_number_after_2013_l3215_321535


namespace pet_store_cages_l3215_321515

/-- Given a pet store scenario with puppies and cages, calculate the number of cages used. -/
theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : puppies_per_cage = 2)
  : (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end pet_store_cages_l3215_321515


namespace fiftieth_term_is_ten_l3215_321587

def sequence_term (n : ℕ) : ℕ := 
  Nat.sqrt (2 * n + 1/4 : ℚ).ceil.toNat + 1

theorem fiftieth_term_is_ten : sequence_term 50 = 10 := by
  sorry

end fiftieth_term_is_ten_l3215_321587


namespace last_installment_theorem_l3215_321595

/-- Represents the installment payment plan for a TV set. -/
structure TVInstallmentPlan where
  total_price : ℕ
  num_installments : ℕ
  installment_amount : ℕ
  interest_rate : ℚ
  first_installment_at_purchase : Bool

/-- Calculates the value of the last installment in a TV installment plan. -/
def last_installment_value (plan : TVInstallmentPlan) : ℕ :=
  plan.installment_amount

/-- Theorem stating that the last installment value is equal to the regular installment amount. -/
theorem last_installment_theorem (plan : TVInstallmentPlan)
  (h1 : plan.total_price = 10000)
  (h2 : plan.num_installments = 20)
  (h3 : plan.installment_amount = 1000)
  (h4 : plan.interest_rate = 6 / 100)
  (h5 : plan.first_installment_at_purchase = true) :
  last_installment_value plan = 1000 := by
  sorry

#eval last_installment_value {
  total_price := 10000,
  num_installments := 20,
  installment_amount := 1000,
  interest_rate := 6 / 100,
  first_installment_at_purchase := true
}

end last_installment_theorem_l3215_321595


namespace circle_radius_tangent_to_three_lines_l3215_321514

/-- A circle with center (0, k) where k > 8 is tangent to y = x, y = -x, and y = 8. Its radius is 8√2. -/
theorem circle_radius_tangent_to_three_lines (k : ℝ) (h1 : k > 8) : 
  let center := (0, k)
  let radius := (λ p : ℝ × ℝ ↦ Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2))
  let tangent_to_line := (λ l : ℝ × ℝ → Prop ↦ ∃ p, l p ∧ radius p = radius center)
  tangent_to_line (λ p ↦ p.2 = p.1) ∧ 
  tangent_to_line (λ p ↦ p.2 = -p.1) ∧
  tangent_to_line (λ p ↦ p.2 = 8) →
  radius center = 8 * Real.sqrt 2 := by
sorry

end circle_radius_tangent_to_three_lines_l3215_321514


namespace parents_disagree_tuition_increase_l3215_321512

theorem parents_disagree_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 20 / 100) : 
  total_parents - (total_parents * agree_percentage).floor = 640 := by
sorry

end parents_disagree_tuition_increase_l3215_321512


namespace g_is_even_l3215_321562

open Real

/-- A function F is odd if F(-x) = -F(x) for all x -/
def IsOdd (F : ℝ → ℝ) : Prop := ∀ x, F (-x) = -F x

/-- A function G is even if G(-x) = G(x) for all x -/
def IsEven (G : ℝ → ℝ) : Prop := ∀ x, G (-x) = G x

/-- Given a > 0, a ≠ 1, and F is an odd function, prove that G(x) = F(x) * (1 / (a^x - 1) + 1/2) is an even function -/
theorem g_is_even (a : ℝ) (ha : a > 0) (hna : a ≠ 1) (F : ℝ → ℝ) (hF : IsOdd F) :
  IsEven (fun x ↦ F x * (1 / (a^x - 1) + 1/2)) := by
  sorry

end g_is_even_l3215_321562


namespace imaginary_part_of_z_l3215_321592

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l3215_321592


namespace intersection_M_N_l3215_321552

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_M_N_l3215_321552


namespace complex_number_location_l3215_321564

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + i) / i
  z = 1 - i ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_location_l3215_321564


namespace range_of_squared_plus_linear_l3215_321576

theorem range_of_squared_plus_linear (a b : ℝ) (h1 : a < -2) (h2 : b > 4) :
  a^2 + b > 8 := by sorry

end range_of_squared_plus_linear_l3215_321576


namespace circle_equation_correct_l3215_321569

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The specific circle we're considering -/
def specific_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, specific_circle x y ↔ circle_equation x y 3 (-1) 4 :=
by sorry

end circle_equation_correct_l3215_321569


namespace rabbit_count_l3215_321506

/-- Given a cage with chickens and rabbits, prove that the number of rabbits is 31 -/
theorem rabbit_count (total_heads : ℕ) (r c : ℕ) : 
  total_heads = 51 →
  r + c = total_heads →
  4 * r = 3 * (2 * c) + 4 →
  r = 31 := by
  sorry

end rabbit_count_l3215_321506


namespace delivery_driver_net_pay_l3215_321524

/-- Calculates the net rate of pay for a delivery driver --/
theorem delivery_driver_net_pay 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) 
  (gasoline_price : ℝ) 
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : earnings_per_mile = 0.60)
  (h5 : gasoline_price = 2.50) : 
  (earnings_per_mile * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_price) / travel_time = 25 := by
  sorry

#check delivery_driver_net_pay

end delivery_driver_net_pay_l3215_321524


namespace function_composition_equals_log_l3215_321540

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1/2 * x - 1/2 else Real.log x

theorem function_composition_equals_log (a : ℝ) :
  (f (f a) = Real.log (f a)) ↔ a ∈ Set.Ici (Real.exp 1) :=
sorry

end function_composition_equals_log_l3215_321540


namespace product_digit_sum_l3215_321597

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  thousands_digit product + units_digit product = 3 := by sorry

end product_digit_sum_l3215_321597


namespace xavier_speed_increase_time_l3215_321523

/-- Represents the journey of Xavier from p to q -/
structure Journey where
  initialSpeed : ℝ  -- Initial speed in km/h
  speedIncrease : ℝ  -- Speed increase in km/h
  totalDistance : ℝ  -- Total distance in km
  totalTime : ℝ  -- Total time in hours

/-- Calculates the time at which Xavier increases his speed -/
def timeOfSpeedIncrease (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that Xavier increases his speed after 24 minutes -/
theorem xavier_speed_increase_time (j : Journey) 
  (h1 : j.initialSpeed = 50)
  (h2 : j.speedIncrease = 10)
  (h3 : j.totalDistance = 52)
  (h4 : j.totalTime = 48 / 60) : 
  timeOfSpeedIncrease j = 24 / 60 := by
  sorry

end xavier_speed_increase_time_l3215_321523


namespace rationalize_denominator_l3215_321537

theorem rationalize_denominator :
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end rationalize_denominator_l3215_321537


namespace kyler_wins_two_l3215_321580

/-- Represents a chess player --/
inductive Player
| Peter
| Emma
| Kyler

/-- Represents the number of games won and lost by a player --/
structure GameRecord where
  player : Player
  wins : ℕ
  losses : ℕ

/-- The total number of games in the tournament --/
def totalGames : ℕ := 6

theorem kyler_wins_two (peter_record : GameRecord) (emma_record : GameRecord) (kyler_record : GameRecord) :
  peter_record.player = Player.Peter ∧
  peter_record.wins = 5 ∧
  peter_record.losses = 4 ∧
  emma_record.player = Player.Emma ∧
  emma_record.wins = 2 ∧
  emma_record.losses = 5 ∧
  kyler_record.player = Player.Kyler ∧
  kyler_record.losses = 4 →
  kyler_record.wins = 2 := by
  sorry

end kyler_wins_two_l3215_321580


namespace angle_measure_proof_l3215_321596

theorem angle_measure_proof (A B : ℝ) : 
  (A = B ∨ A + B = 180) →  -- Parallel sides condition
  A = 3 * B - 20 →         -- Relationship between A and B
  A = 10 ∨ A = 130 :=      -- Conclusion
by sorry

end angle_measure_proof_l3215_321596


namespace kaleb_net_profit_l3215_321505

/-- Calculates the net profit for Kaleb's lawn mowing business --/
def net_profit (small_charge medium_charge large_charge : ℕ)
                (spring_small spring_medium spring_large : ℕ)
                (summer_small summer_medium summer_large : ℕ)
                (fuel_expense supply_cost : ℕ) : ℕ :=
  let spring_earnings := small_charge * spring_small + medium_charge * spring_medium + large_charge * spring_large
  let summer_earnings := small_charge * summer_small + medium_charge * summer_medium + large_charge * summer_large
  let total_earnings := spring_earnings + summer_earnings
  let total_lawns := spring_small + spring_medium + spring_large + summer_small + summer_medium + summer_large
  let total_expenses := fuel_expense * total_lawns + supply_cost
  total_earnings - total_expenses

theorem kaleb_net_profit :
  net_profit 10 20 30 2 3 1 10 8 5 2 60 = 402 := by
  sorry

end kaleb_net_profit_l3215_321505


namespace function_inequality_and_logarithm_comparison_l3215_321563

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- State the theorem
theorem function_inequality_and_logarithm_comparison (m : ℝ) 
  (h : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) : 
  m > 0 ∧ Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end function_inequality_and_logarithm_comparison_l3215_321563


namespace symmetric_point_of_P_l3215_321574

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Given point P -/
def P : Point3D := ⟨3, 1, 5⟩

/-- Function to find the symmetric point about the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

/-- Theorem: The point symmetric to P(3,1,5) about the origin is (3,-1,-5) -/
theorem symmetric_point_of_P :
  symmetricPoint P = Point3D.mk 3 (-1) (-5) := by
  sorry

end symmetric_point_of_P_l3215_321574


namespace stratified_sampling_theorem_l3215_321573

/-- Represents a teacher with their name and number of created questions -/
structure Teacher where
  name : String
  questions : ℕ

/-- Represents the result of stratified sampling -/
structure SamplingResult where
  wu : ℕ
  wang : ℕ
  zhang : ℕ

/-- Calculates the number of questions selected for each teacher in stratified sampling -/
def stratifiedSampling (teachers : List Teacher) (totalSamples : ℕ) : SamplingResult :=
  sorry

/-- Calculates the probability of selecting at least one question from a specific teacher -/
def probabilityAtLeastOne (samplingResult : SamplingResult) (teacherQuestions : ℕ) (selectionSize : ℕ) : ℚ :=
  sorry

theorem stratified_sampling_theorem (wu wang zhang : Teacher) (h1 : wu.questions = 350) (h2 : wang.questions = 700) (h3 : zhang.questions = 1050) :
  let teachers := [wu, wang, zhang]
  let result := stratifiedSampling teachers 6
  result.wu = 1 ∧ result.wang = 2 ∧ result.zhang = 3 ∧
  probabilityAtLeastOne result result.wang 2 = 3/5 := by
  sorry

end stratified_sampling_theorem_l3215_321573


namespace bowling_team_new_average_l3215_321534

def bowling_team_average (original_players : ℕ) (original_average : ℚ) (new_player1_weight : ℚ) (new_player2_weight : ℚ) : ℚ :=
  let original_total_weight := original_players * original_average
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

theorem bowling_team_new_average :
  bowling_team_average 7 121 110 60 = 113 := by
  sorry

end bowling_team_new_average_l3215_321534


namespace pizza_combinations_l3215_321520

/-- The number of available toppings -/
def num_toppings : ℕ := 8

/-- The number of forbidden topping combinations -/
def num_forbidden : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of possible one- and two-topping pizzas -/
def total_pizzas : ℕ := num_toppings + choose num_toppings 2 - num_forbidden

theorem pizza_combinations :
  total_pizzas = 35 :=
sorry

end pizza_combinations_l3215_321520


namespace hcf_problem_l3215_321503

theorem hcf_problem (x y : ℕ+) 
  (h1 : Nat.lcm x y = 560) 
  (h2 : x * y = 42000) : 
  Nat.gcd x y = 75 := by
sorry

end hcf_problem_l3215_321503


namespace hyperbola_properties_l3215_321551

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the asymptotic lines
def asymptotic_lines (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_properties
  (a b : ℝ)
  (h_positive : a > 0 ∧ b > 0)
  (h_point : hyperbola a b (-2) (Real.sqrt 6))
  (h_asymptotic : ∀ x y, hyperbola a b x y → asymptotic_lines x y) :
  -- 1) The equation of C is x^2 - y^2/2 = 1
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2/2 = 1) ∧
  -- 2) P cannot be the midpoint of any chord AB of C
  (∀ A B : ℝ × ℝ,
    (hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2) →
    (∃ k : ℝ, A.2 - point_P.2 = k * (A.1 - point_P.1) ∧
              B.2 - point_P.2 = k * (B.1 - point_P.1)) →
    point_P ≠ ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end hyperbola_properties_l3215_321551


namespace division_remainder_l3215_321549

theorem division_remainder :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 136 →
    divisor = 15 →
    quotient = 9 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end division_remainder_l3215_321549


namespace marbles_combination_l3215_321539

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marbles_combination :
  choose 10 4 = 210 := by sorry

end marbles_combination_l3215_321539


namespace integer_triple_product_sum_l3215_321508

theorem integer_triple_product_sum (a b c : ℤ) : 
  (a * b * c = 4 * (a + b + c) ∧ c = 2 * (a + b)) ↔ 
  ((a = 1 ∧ b = 6 ∧ c = 14) ∨ 
   (a = -1 ∧ b = -6 ∧ c = -14) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 10) ∨ 
   (a = -2 ∧ b = -3 ∧ c = -10) ∨ 
   (b = -a ∧ c = 0)) := by
sorry

end integer_triple_product_sum_l3215_321508


namespace unsold_bars_unsold_bars_correct_l3215_321530

/-- Proves the number of unsold chocolate bars given the total number of bars,
    the cost per bar, and the total money made from sold bars. -/
theorem unsold_bars (total_bars : ℕ) (cost_per_bar : ℕ) (money_made : ℕ) : ℕ :=
  total_bars - (money_made / cost_per_bar)

#check unsold_bars 7 3 9 = 4

theorem unsold_bars_correct :
  unsold_bars 7 3 9 = 4 := by sorry

end unsold_bars_unsold_bars_correct_l3215_321530


namespace rounding_comparison_l3215_321521

theorem rounding_comparison (a b : ℝ) : 
  (2.35 ≤ a ∧ a ≤ 2.44) → 
  (2.395 ≤ b ∧ b ≤ 2.404) → 
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x = y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x > y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x < y) :=
by sorry

end rounding_comparison_l3215_321521


namespace frog_population_estimate_l3215_321519

/-- Estimates the number of frogs in a pond based on capture-recapture data and population changes --/
theorem frog_population_estimate (tagged_april : ℕ) (caught_august : ℕ) (tagged_recaptured : ℕ)
  (left_pond_percent : ℚ) (new_frogs_percent : ℚ)
  (h1 : tagged_april = 100)
  (h2 : caught_august = 90)
  (h3 : tagged_recaptured = 5)
  (h4 : left_pond_percent = 30 / 100)
  (h5 : new_frogs_percent = 35 / 100) :
  let april_frogs_in_august := caught_august * (1 - new_frogs_percent)
  let estimated_april_population := (tagged_april * april_frogs_in_august) / tagged_recaptured
  estimated_april_population = 1180 := by
sorry

end frog_population_estimate_l3215_321519


namespace base_b_is_four_l3215_321581

theorem base_b_is_four : 
  ∃ (b : ℕ), 
    b > 0 ∧ 
    (b - 1) * (b - 1) * b = 72 ∧ 
    b = 4 := by
  sorry

end base_b_is_four_l3215_321581


namespace largest_five_digit_with_product_2772_l3215_321548

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_2772 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 2772 → n ≤ 98721 :=
by sorry

end largest_five_digit_with_product_2772_l3215_321548


namespace local_max_condition_l3215_321501

/-- The function f(x) = x(x-m)² has a local maximum at x = 1 if and only if m = 3 -/
theorem local_max_condition (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), x * (x - m)^2 ≤ 1 * (1 - m)^2) ↔ m = 3 :=
by sorry

end local_max_condition_l3215_321501


namespace custom_op_theorem_l3215_321510

/-- Custom operation ã — -/
def custom_op (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem custom_op_theorem :
  ∃ X : ℝ, X + 2 * (custom_op 1 3) = 7 →
  3 * (custom_op 1 2) = 12 * 1 - 18 := by
sorry

end custom_op_theorem_l3215_321510


namespace roberts_reading_l3215_321550

/-- Given Robert's reading rate and book length, calculate the maximum number of complete books he can read in a given time. -/
theorem roberts_reading (reading_rate : ℕ) (book_length : ℕ) (available_time : ℕ) :
  reading_rate > 0 →
  book_length > 0 →
  available_time > 0 →
  reading_rate = 120 →
  book_length = 360 →
  available_time = 8 →
  (available_time * reading_rate) / book_length = 2 :=
by sorry

end roberts_reading_l3215_321550


namespace fraction_expression_equality_l3215_321559

theorem fraction_expression_equality : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end fraction_expression_equality_l3215_321559


namespace crow_eating_time_l3215_321538

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def eating_time (fraction : ℚ) : ℚ :=
  7.5 / (1/4) * fraction

theorem crow_eating_time :
  eating_time (1/5) = 6 := by
  sorry

end crow_eating_time_l3215_321538


namespace fq_length_l3215_321584

/-- Represents a right triangle with a tangent circle -/
structure RightTriangleWithCircle where
  /-- Length of the hypotenuse -/
  df : ℝ
  /-- Length of one leg -/
  de : ℝ
  /-- Point where the circle meets the hypotenuse -/
  q : ℝ
  /-- The hypotenuse is √85 -/
  hyp_length : df = Real.sqrt 85
  /-- One leg is 7 -/
  leg_length : de = 7
  /-- The circle is tangent to both legs -/
  circle_tangent : True

/-- The length of FQ in the given configuration is 6 -/
theorem fq_length (t : RightTriangleWithCircle) : t.df - t.q = 6 := by
  sorry

end fq_length_l3215_321584
