import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3698_369821

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3698_369821


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3698_369870

/-- The imaginary part of 1 / (1 + i) is -1/2 -/
theorem imaginary_part_of_z (z : ℂ) : z = 1 / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3698_369870


namespace NUMINAMATH_CALUDE_incorrect_relation_l3698_369875

theorem incorrect_relation (a b : ℝ) (h : a > b) : ∃ c : ℝ, ¬(a * c^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_relation_l3698_369875


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3698_369826

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 15)
  (h2 : boys = 25)
  (h3 : called_back = 7) :
  girls + boys - called_back = 33 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3698_369826


namespace NUMINAMATH_CALUDE_root_transformation_l3698_369865

theorem root_transformation (a b c d : ℂ) : 
  (a^4 - 2*a - 6 = 0) ∧ 
  (b^4 - 2*b - 6 = 0) ∧ 
  (c^4 - 2*c - 6 = 0) ∧ 
  (d^4 - 2*d - 6 = 0) →
  ∃ (y₁ y₂ y₃ y₄ : ℂ), 
    y₁ = 2*(a + b + c)/d^3 ∧
    y₂ = 2*(a + b + d)/c^3 ∧
    y₃ = 2*(a + c + d)/b^3 ∧
    y₄ = 2*(b + c + d)/a^3 ∧
    (2*y₁^4 - 2*y₁ + 48 = 0) ∧
    (2*y₂^4 - 2*y₂ + 48 = 0) ∧
    (2*y₃^4 - 2*y₃ + 48 = 0) ∧
    (2*y₄^4 - 2*y₄ + 48 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l3698_369865


namespace NUMINAMATH_CALUDE_johns_bookshop_l3698_369834

/-- The total number of books sold over 5 days -/
def total_sold : ℕ := 280

/-- The percentage of books that were not sold -/
def percent_not_sold : ℚ := 54.83870967741935

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 620

theorem johns_bookshop :
  initial_books = total_sold / ((100 - percent_not_sold) / 100) := by sorry

end NUMINAMATH_CALUDE_johns_bookshop_l3698_369834


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3698_369811

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3698_369811


namespace NUMINAMATH_CALUDE_money_division_l3698_369824

theorem money_division (total : ℕ) (p q r : ℕ) (h1 : p + q + r = total) (h2 : p = 3 * (total / 22)) (h3 : q = 7 * (total / 22)) (h4 : r = 12 * (total / 22)) (h5 : r - q = 5500) : q - p = 4400 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3698_369824


namespace NUMINAMATH_CALUDE_quadrilateral_tile_exists_l3698_369892

/-- A quadrilateral tile with angles measured in degrees -/
structure QuadTile where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- The property that six tiles meet at a vertex -/
def sixTilesMeet (t : QuadTile) : Prop :=
  ∃ (i : Fin 4), t.angle1 * (i.val : ℝ) + t.angle2 * ((4 - i).val : ℝ) = 360

/-- The sum of angles in a quadrilateral is 360° -/
def validQuadrilateral (t : QuadTile) : Prop :=
  t.angle1 + t.angle2 + t.angle3 + t.angle4 = 360

/-- The main theorem: there exists a quadrilateral tile with the specified angles -/
theorem quadrilateral_tile_exists : ∃ (t : QuadTile), 
  t.angle1 = 45 ∧ t.angle2 = 60 ∧ t.angle3 = 105 ∧ t.angle4 = 150 ∧
  sixTilesMeet t ∧ validQuadrilateral t :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_tile_exists_l3698_369892


namespace NUMINAMATH_CALUDE_range_of_a_l3698_369853

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 2*a*x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3698_369853


namespace NUMINAMATH_CALUDE_go_match_results_l3698_369849

structure GoMatch where
  redWinProb : ℝ
  mk_prob_valid : 0 ≤ redWinProb ∧ redWinProb ≤ 1

def RedTeam := Fin 3 → GoMatch

def atLeastTwoWins (team : RedTeam) : ℝ :=
  sorry

def expectedWins (team : RedTeam) : ℝ :=
  sorry

theorem go_match_results (team : RedTeam) 
  (h1 : team 0 = ⟨0.6, sorry⟩) 
  (h2 : team 1 = ⟨0.5, sorry⟩)
  (h3 : team 2 = ⟨0.5, sorry⟩) :
  atLeastTwoWins team = 0.55 ∧ expectedWins team = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_go_match_results_l3698_369849


namespace NUMINAMATH_CALUDE_max_boxes_arrangement_l3698_369872

/-- A Box represents a rectangle in the plane with sides parallel to coordinate axes. -/
structure Box where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_positive : x₁ < x₂ ∧ y₁ < y₂

/-- Two boxes intersect if they have a common point. -/
def intersect (b₁ b₂ : Box) : Prop :=
  ¬(b₁.x₂ ≤ b₂.x₁ ∨ b₂.x₂ ≤ b₁.x₁ ∨ b₁.y₂ ≤ b₂.y₁ ∨ b₂.y₂ ≤ b₁.y₁)

/-- A valid arrangement of n boxes satisfies the intersection condition. -/
def valid_arrangement (n : ℕ) (boxes : Fin n → Box) : Prop :=
  ∀ i j : Fin n, intersect (boxes i) (boxes j) ↔ (i.val + 1) % n ≠ j.val ∧ (i.val + n - 1) % n ≠ j.val

/-- The main theorem: The maximum number of boxes in a valid arrangement is 6. -/
theorem max_boxes_arrangement :
  (∃ (boxes : Fin 6 → Box), valid_arrangement 6 boxes) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (boxes : Fin n → Box), valid_arrangement n boxes) :=
sorry

end NUMINAMATH_CALUDE_max_boxes_arrangement_l3698_369872


namespace NUMINAMATH_CALUDE_max_height_foldable_triangle_l3698_369845

/-- The maximum height of a foldable table constructed from a triangle --/
theorem max_height_foldable_triangle (PQ QR PR : ℝ) (h_PQ : PQ = 24) (h_QR : QR = 32) (h_PR : PR = 40) :
  let s := (PQ + QR + PR) / 2
  let A := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  let h_p := 2 * A / QR
  let h_q := 2 * A / PR
  let h_r := 2 * A / PQ
  let h' := min (h_p * h_q / (h_p + h_q)) (min (h_q * h_r / (h_q + h_r)) (h_r * h_p / (h_r + h_p)))
  h' = 48 * Real.sqrt 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_max_height_foldable_triangle_l3698_369845


namespace NUMINAMATH_CALUDE_housing_price_growth_l3698_369864

/-- Proves that the equation relating initial housing price, final housing price, 
    and annual growth rate over two years is correct. -/
theorem housing_price_growth (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 5500)
  (h_final : final_price = 7000) :
  initial_price * (1 + x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_housing_price_growth_l3698_369864


namespace NUMINAMATH_CALUDE_kitchen_tile_size_l3698_369822

/-- Given a rectangular kitchen floor and the number of tiles needed, 
    calculate the size of each tile. -/
theorem kitchen_tile_size 
  (length : ℕ) 
  (width : ℕ) 
  (num_tiles : ℕ) 
  (h1 : length = 48) 
  (h2 : width = 72) 
  (h3 : num_tiles = 96) : 
  (length * width) / num_tiles = 36 := by
  sorry

#check kitchen_tile_size

end NUMINAMATH_CALUDE_kitchen_tile_size_l3698_369822


namespace NUMINAMATH_CALUDE_largest_number_proof_l3698_369899

theorem largest_number_proof (a b c d : ℕ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3698_369899


namespace NUMINAMATH_CALUDE_product_local_abs_value_l3698_369856

-- Define the complex number
def z : ℂ := 564823 + 3*Complex.I

-- Define the digit of interest
def digit : ℕ := 4

-- Define the local value of the digit in the complex number
def local_value : ℕ := 4000

-- Define the absolute value of the digit
def abs_digit : ℕ := 4

-- Theorem to prove
theorem product_local_abs_value : 
  local_value * abs_digit = 16000 := by sorry

end NUMINAMATH_CALUDE_product_local_abs_value_l3698_369856


namespace NUMINAMATH_CALUDE_triangle_properties_l3698_369832

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1/3) :
  t.B = π/6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l3698_369832


namespace NUMINAMATH_CALUDE_unique_solution_power_sum_square_l3698_369888

theorem unique_solution_power_sum_square :
  ∃! (x y z : ℕ+), 2^(x.val) + 3^(y.val) = z.val^2 ∧ x = 4 ∧ y = 2 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_sum_square_l3698_369888


namespace NUMINAMATH_CALUDE_circle_square_area_l3698_369889

/-- A circle described by the equation 2x^2 = -2y^2 + 8x - 8y + 28 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 = -2 * p.2^2 + 8 * p.1 - 8 * p.2 + 28}

/-- The square that circumscribes the circle with sides parallel to the axes -/
def CircumscribingSquare (c : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), (x, y) ∈ c ∧ 
    (p.1 = x ∨ p.1 = -x) ∧ (p.2 = y ∨ p.2 = -y)}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_square_area : area (CircumscribingSquare Circle) = 88 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_l3698_369889


namespace NUMINAMATH_CALUDE_work_completion_time_l3698_369896

/-- The number of laborers originally employed by the contractor -/
def original_laborers : ℚ := 17.5

/-- The number of absent laborers -/
def absent_laborers : ℕ := 7

/-- The number of days it took the remaining laborers to complete the work -/
def actual_days : ℕ := 10

/-- The original number of days the work was supposed to be completed in -/
def original_days : ℚ := (original_laborers - absent_laborers : ℚ) * actual_days / original_laborers

theorem work_completion_time : original_days = 6 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3698_369896


namespace NUMINAMATH_CALUDE_equation_proof_l3698_369828

-- Define the variables and the given equation
theorem equation_proof (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * Real.pi) = Q) :
  8 * (10 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3698_369828


namespace NUMINAMATH_CALUDE_log_power_sum_l3698_369827

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 8) (h2 : b = Real.log 27) :
  (5 : ℝ) ^ (a / b) + 2 ^ (b / a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_power_sum_l3698_369827


namespace NUMINAMATH_CALUDE_balloon_difference_l3698_369885

theorem balloon_difference (allan_initial : ℕ) (allan_bought : ℕ) (jake_balloons : ℕ)
  (h1 : allan_initial = 2)
  (h2 : allan_bought = 3)
  (h3 : jake_balloons = 6) :
  jake_balloons - (allan_initial + allan_bought) = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3698_369885


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3698_369880

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3698_369880


namespace NUMINAMATH_CALUDE_childrens_admission_fee_l3698_369882

/-- Proves that the children's admission fee is $1.50 given the problem conditions -/
theorem childrens_admission_fee (total_people : ℕ) (total_fees : ℚ) (num_children : ℕ) (adult_fee : ℚ) :
  total_people = 315 →
  total_fees = 810 →
  num_children = 180 →
  adult_fee = 4 →
  ∃ (child_fee : ℚ),
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fees ∧
    child_fee = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_childrens_admission_fee_l3698_369882


namespace NUMINAMATH_CALUDE_common_sum_in_square_matrix_l3698_369881

theorem common_sum_in_square_matrix : 
  let n : ℕ := 36
  let a : ℤ := -15
  let l : ℤ := 20
  let total_sum : ℤ := n * (a + l) / 2
  let matrix_size : ℕ := 6
  total_sum / matrix_size = 15 := by sorry

end NUMINAMATH_CALUDE_common_sum_in_square_matrix_l3698_369881


namespace NUMINAMATH_CALUDE_relationship_functions_l3698_369803

-- Define the relationships
def relationA (x : ℝ) : ℝ := 180 - x
def relationB (x : ℝ) : ℝ := 60 + 3 * x
def relationC (x : ℝ) : ℝ := x ^ 2
def relationD (x : ℝ) : Set ℝ := {y | y ^ 2 = x ∧ x ≥ 0}

-- Theorem stating that A, B, and C are functions, while D is not
theorem relationship_functions :
  (∀ x : ℝ, ∃! y : ℝ, y = relationA x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationB x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationC x) ∧
  ¬(∀ x : ℝ, ∃! y : ℝ, y ∈ relationD x) :=
by sorry

end NUMINAMATH_CALUDE_relationship_functions_l3698_369803


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3698_369846

/-- The function f(x) = -x^2 + x + m + 2 -/
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

/-- The solution set of f(x) ≥ |x| -/
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | f x m ≥ |x|}

/-- The set of integers in the solution set -/
def integer_solutions (m : ℝ) : Set ℤ := {i : ℤ | (i : ℝ) ∈ solution_set m}

theorem unique_integer_solution (m : ℝ) :
  (∃! (i : ℤ), (i : ℝ) ∈ solution_set m) → -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3698_369846


namespace NUMINAMATH_CALUDE_men_who_left_hostel_l3698_369843

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_who_left_hostel 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 28)
  (h3 : final_days = 35)
  (h4 : initial_men * initial_days = (initial_men - men_who_left) * final_days) :
  men_who_left = 50 := by
  sorry

#check men_who_left_hostel

end NUMINAMATH_CALUDE_men_who_left_hostel_l3698_369843


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l3698_369818

theorem cube_sum_divisibility (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (9 ∣ (a₁^3 + a₂^3 + a₃^3 + a₄^3 + a₅^3)) → (3 ∣ (a₁ * a₂ * a₃ * a₄ * a₅)) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l3698_369818


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l3698_369807

theorem tan_seventeen_pi_fourths : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l3698_369807


namespace NUMINAMATH_CALUDE_equation_solutions_l3698_369800

theorem equation_solutions :
  (∀ x, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x, x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3698_369800


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l3698_369867

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let basketballs_per_player : ℕ := 11
  num_players * basketballs_per_player = 242 :=
by sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l3698_369867


namespace NUMINAMATH_CALUDE_cosine_is_periodic_l3698_369861

-- Define a type for functions from ℝ to ℝ
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be periodic
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define what it means for a function to be trigonometric
def IsTrigonometric (f : RealFunction) : Prop :=
  -- This is a placeholder definition
  True

-- State the theorem
theorem cosine_is_periodic :
  (∀ f : RealFunction, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (λ x : ℝ => Real.cos x) →
  IsPeriodic (λ x : ℝ => Real.cos x) :=
by
  sorry

end NUMINAMATH_CALUDE_cosine_is_periodic_l3698_369861


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l3698_369815

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = (3 - x) / 2
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x

-- Theorem stating that the equation represents three distinct lines
-- that do not all intersect at a single point
theorem equation_represents_three_lines :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p2.1 p2.2 ∧ line3 p3.1 p3.2) ∧
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :=
  sorry


end NUMINAMATH_CALUDE_equation_represents_three_lines_l3698_369815


namespace NUMINAMATH_CALUDE_car_distance_l3698_369847

/-- Given a total distance of 40 kilometers, if 1/4 of the distance is traveled by foot
    and 1/2 of the distance is traveled by bus, then the remaining distance traveled
    by car is 10 kilometers. -/
theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) 
    (h1 : total_distance = 40)
    (h2 : foot_fraction = 1/4)
    (h3 : bus_fraction = 1/2) :
    total_distance - (foot_fraction * total_distance) - (bus_fraction * total_distance) = 10 := by
  sorry


end NUMINAMATH_CALUDE_car_distance_l3698_369847


namespace NUMINAMATH_CALUDE_not_p_and_not_q_true_l3698_369830

theorem not_p_and_not_q_true (p q : Prop)
  (h1 : ¬(p ∧ q))
  (h2 : ¬(p ∨ q)) :
  (¬p ∧ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_true_l3698_369830


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3698_369897

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 →
    (50 * x - 42) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -6264 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3698_369897


namespace NUMINAMATH_CALUDE_jogger_train_distance_l3698_369884

/-- Proves that a jogger is 200 meters ahead of a train's engine given specific conditions --/
theorem jogger_train_distance (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  train_length = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time = train_length + 200 :=
by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l3698_369884


namespace NUMINAMATH_CALUDE_white_bread_count_l3698_369858

/-- The number of loaves of white bread bought each week -/
def white_bread_loaves : ℕ := sorry

/-- The cost of a loaf of white bread -/
def white_bread_cost : ℚ := 7/2

/-- The cost of a baguette -/
def baguette_cost : ℚ := 3/2

/-- The number of sourdough loaves bought each week -/
def sourdough_loaves : ℕ := 2

/-- The cost of a loaf of sourdough bread -/
def sourdough_cost : ℚ := 9/2

/-- The cost of an almond croissant -/
def croissant_cost : ℚ := 2

/-- The number of weeks -/
def weeks : ℕ := 4

/-- The total amount spent over all weeks -/
def total_spent : ℚ := 78

theorem white_bread_count :
  white_bread_loaves = 2 ∧
  (white_bread_loaves : ℚ) * white_bread_cost +
  baguette_cost +
  (sourdough_loaves : ℚ) * sourdough_cost +
  croissant_cost =
  total_spent / (weeks : ℚ) :=
sorry

end NUMINAMATH_CALUDE_white_bread_count_l3698_369858


namespace NUMINAMATH_CALUDE_equation_equivalence_l3698_369886

theorem equation_equivalence (a b c : ℝ) :
  (1 / (a + b) + 1 / (b + c) = 2 / (c + a)) ↔ (2 * b^2 = a^2 + c^2) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3698_369886


namespace NUMINAMATH_CALUDE_max_value_theorem_l3698_369855

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ (16 * Real.sqrt 3 + 2 * Real.sqrt 33) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3698_369855


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3698_369842

theorem intersection_of_sets (P Q : Set ℝ) : 
  (P = {y : ℝ | ∃ x : ℝ, y = x + 1}) → 
  (Q = {y : ℝ | ∃ x : ℝ, y = 1 - x}) → 
  P ∩ Q = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3698_369842


namespace NUMINAMATH_CALUDE_justin_pencils_l3698_369804

theorem justin_pencils (total_pencils sabrina_pencils : ℕ) : 
  total_pencils = 50 →
  sabrina_pencils = 14 →
  total_pencils - sabrina_pencils > 2 * sabrina_pencils →
  (total_pencils - sabrina_pencils) - 2 * sabrina_pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_justin_pencils_l3698_369804


namespace NUMINAMATH_CALUDE_problem_solution_l3698_369829

noncomputable section

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem problem_solution (a : ℝ) :
  (∃ x, x ∈ A a) ∧ (∃ x, x ∈ B a) →
  (a = 1/2 → (U \ B a) ∩ A a = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  (A a ⊆ B a ↔ a ∈ Set.Icc (-1/2) (1/3) ∪ Set.Ioc (1/3) ((3 - Real.sqrt 5) / 2)) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3698_369829


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l3698_369848

/-- Given points A(0,6), B(0,0), C(8,0), and D the midpoint of AB, 
    prove that the sum of the slope and y-intercept of the line passing through C and D is 21/8 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → B = (0, 0) → C = (8, 0) → D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 21 / 8 := by
sorry


end NUMINAMATH_CALUDE_slope_intercept_sum_l3698_369848


namespace NUMINAMATH_CALUDE_factorizable_polynomial_count_l3698_369816

theorem factorizable_polynomial_count : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2000) ∧
    (∀ n ∈ S, ∃ a b : ℤ, ∀ x, x^2 - 3*x - n = (x - a) * (x - b)) ∧
    (∀ n : ℕ, 1 ≤ n → n ≤ 2000 → 
      (∃ a b : ℤ, ∀ x, x^2 - 3*x - n = (x - a) * (x - b)) → n ∈ S) ∧
    S.card = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_factorizable_polynomial_count_l3698_369816


namespace NUMINAMATH_CALUDE_probability_neither_cake_nor_muffin_l3698_369817

def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_buyers : ℕ := 16

theorem probability_neither_cake_nor_muffin :
  let buyers_of_at_least_one := cake_buyers + muffin_buyers - both_buyers
  let buyers_of_neither := total_buyers - buyers_of_at_least_one
  (buyers_of_neither : ℚ) / total_buyers = 26 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_cake_nor_muffin_l3698_369817


namespace NUMINAMATH_CALUDE_anniversary_sale_cost_l3698_369874

/-- The cost of the purchase during the anniversary sale -/
def total_cost (original_ice_cream_price sale_discount juice_price_per_5 ice_cream_tubs juice_cans : ℚ) : ℚ :=
  (ice_cream_tubs * (original_ice_cream_price - sale_discount)) + 
  (juice_cans / 5 * juice_price_per_5)

/-- Theorem stating that the total cost of the purchase is $24 -/
theorem anniversary_sale_cost : 
  total_cost 12 2 2 2 10 = 24 := by sorry

end NUMINAMATH_CALUDE_anniversary_sale_cost_l3698_369874


namespace NUMINAMATH_CALUDE_sequence_problem_l3698_369879

def D (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem sequence_problem (A : ℕ → ℝ) 
  (h1 : ∀ n, D (D A) n = 1) 
  (h2 : A 19 = 0) 
  (h3 : A 92 = 0) : 
  A 1 = 819 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3698_369879


namespace NUMINAMATH_CALUDE_hearts_clubs_equal_prob_l3698_369814

/-- Represents the suit of a playing card -/
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

/-- Represents the rank of a playing card -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  rank : Rank

/-- Represents a standard deck of playing cards -/
def standardDeck : Finset Card := sorry

/-- The number of cards in a standard deck -/
def deckSize : Nat := 52

/-- The number of cards of each suit in a standard deck -/
def suitCount : Nat := 13

/-- The probability of drawing a card of a specific suit from a standard deck -/
def probSuit (s : Suit) : ℚ :=
  suitCount / deckSize

theorem hearts_clubs_equal_prob :
  probSuit Suit.Hearts = probSuit Suit.Clubs := by
  sorry

end NUMINAMATH_CALUDE_hearts_clubs_equal_prob_l3698_369814


namespace NUMINAMATH_CALUDE_hyperbola_focus_implies_m_l3698_369801

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 / m - x^2 / 9 = 1

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (0, 5)

/-- Theorem: If F(0,5) is a focus of the hyperbola y^2/m - x^2/9 = 1, then m = 16 -/
theorem hyperbola_focus_implies_m (m : ℝ) :
  (∀ x y, hyperbola_equation x y m → (x - focus.1)^2 + (y - focus.2)^2 = (x + focus.1)^2 + (y - focus.2)^2) →
  m = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_implies_m_l3698_369801


namespace NUMINAMATH_CALUDE_product_of_solutions_l3698_369877

theorem product_of_solutions (x : ℝ) : 
  (25 = 3 * x^2 + 10 * x) → 
  (∃ α β : ℝ, (3 * α^2 + 10 * α = 25) ∧ (3 * β^2 + 10 * β = 25) ∧ (α * β = -25/3)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3698_369877


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3698_369868

theorem unique_solution_exists : ∃! (x y : ℝ), 
  0.75 * x - 0.40 * y = 0.20 * 422.50 ∧ 
  0.30 * x + 0.50 * y = 0.35 * 530 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3698_369868


namespace NUMINAMATH_CALUDE_transformation_maps_correctly_l3698_369862

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a given factor -/
def scale (p : Point) (factor : ℝ) : Point :=
  ⟨p.x * factor, p.y * factor⟩

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_correctly :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨2, 3⟩
  let A' : Point := ⟨3, -6⟩
  let B' : Point := ⟨6, -9⟩
  (scaleAndReflect A 3 = A') ∧ (scaleAndReflect B 3 = B') := by
  sorry

end NUMINAMATH_CALUDE_transformation_maps_correctly_l3698_369862


namespace NUMINAMATH_CALUDE_clara_cookies_sold_l3698_369836

/-- Calculates the total number of cookies sold by Clara -/
def total_cookies_sold (cookies_per_box : Fin 3 → ℕ) (boxes_sold : Fin 3 → ℕ) : ℕ :=
  (cookies_per_box 0) * (boxes_sold 0) + 
  (cookies_per_box 1) * (boxes_sold 1) + 
  (cookies_per_box 2) * (boxes_sold 2)

/-- Proves that Clara sells 3320 cookies in total -/
theorem clara_cookies_sold :
  let cookies_per_box : Fin 3 → ℕ := ![12, 20, 16]
  let boxes_sold : Fin 3 → ℕ := ![50, 80, 70]
  total_cookies_sold cookies_per_box boxes_sold = 3320 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookies_sold_l3698_369836


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3698_369813

-- Define the circle's center
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the conditions for the circle's center
def satisfiesConditions (c : CircleCenter) : Prop :=
  c.x - 2 * c.y = 0 ∧
  3 * c.x - 4 * c.y = 10

-- Theorem statement
theorem circle_center_coordinates :
  ∃ (c : CircleCenter), satisfiesConditions c ∧ c.x = 10 ∧ c.y = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3698_369813


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3698_369893

theorem quadratic_inequality_solution (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ k ∈ Set.Ioo (-8/3) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3698_369893


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3698_369850

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 14) 
  (h2 : x - y = 60) : 
  x = 37 := by sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3698_369850


namespace NUMINAMATH_CALUDE_inequality_solution_quadratic_solution_l3698_369837

-- Part 1: Integer solutions of the inequality
def integer_solutions : Set ℤ :=
  {x : ℤ | -2 ≤ (1 + 2*x) / 3 ∧ (1 + 2*x) / 3 ≤ 2}

theorem inequality_solution :
  integer_solutions = {-3, -2, -1, 0, 1, 2} := by sorry

-- Part 2: Quadratic equation
def quadratic_equation (a b : ℚ) (x : ℚ) : ℚ :=
  a * x^2 + b * x

theorem quadratic_solution (a b : ℚ) :
  (quadratic_equation a b 1 = 0 ∧ quadratic_equation a b 2 = 3) →
  quadratic_equation a b (-2) = 9 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_quadratic_solution_l3698_369837


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l3698_369823

theorem correct_subtraction_result 
  (mistaken_result : ℕ)
  (tens_digit_increase : ℕ)
  (units_digit_increase : ℕ)
  (h1 : mistaken_result = 217)
  (h2 : tens_digit_increase = 3)
  (h3 : units_digit_increase = 4) :
  mistaken_result - (tens_digit_increase * 10 - units_digit_increase) = 191 :=
by sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l3698_369823


namespace NUMINAMATH_CALUDE_sum_of_star_equation_l3698_369866

/-- Custom operation ★ -/
def star (a b : ℕ) : ℕ := a^b + a + b

theorem sum_of_star_equation {a b : ℕ} (ha : a ≥ 2) (hb : b ≥ 2) (heq : star a b = 20) :
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_star_equation_l3698_369866


namespace NUMINAMATH_CALUDE_commission_proof_l3698_369863

/-- Calculates the commission earned from selling a coupe and an SUV --/
def calculate_commission (coupe_price : ℝ) (suv_multiplier : ℝ) (commission_rate : ℝ) : ℝ :=
  let suv_price := coupe_price * suv_multiplier
  let total_sales := coupe_price + suv_price
  total_sales * commission_rate

/-- Proves that the commission earned is $1,800 given the specified conditions --/
theorem commission_proof :
  calculate_commission 30000 2 0.02 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_commission_proof_l3698_369863


namespace NUMINAMATH_CALUDE_area_of_T_is_34_l3698_369812

/-- The area of a "T" shape formed within a rectangle -/
def area_of_T (rectangle_width rectangle_height removed_width removed_height : ℕ) : ℕ :=
  rectangle_width * rectangle_height - removed_width * removed_height

/-- Theorem stating that the area of the "T" shape is 34 square units -/
theorem area_of_T_is_34 :
  area_of_T 10 4 6 1 = 34 := by
  sorry

end NUMINAMATH_CALUDE_area_of_T_is_34_l3698_369812


namespace NUMINAMATH_CALUDE_sum_distances_foci_to_line_l3698_369805

/-- The ellipse C in the xy-plane -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- The left focus of ellipse C -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of ellipse C -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def dist_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- Theorem: The sum of distances from the foci of ellipse C to line l is 2√2 -/
theorem sum_distances_foci_to_line :
  dist_point_to_line F₁ line_l + dist_point_to_line F₂ line_l = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_distances_foci_to_line_l3698_369805


namespace NUMINAMATH_CALUDE_platform_length_l3698_369841

/-- The length of a platform given a train's speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 30 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 150 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3698_369841


namespace NUMINAMATH_CALUDE_sector_area_theorem_l3698_369840

/-- Given a circular sector with central angle θ and arc length l,
    prove that if θ = 2 and l = 2, then the area of the sector is 1. -/
theorem sector_area_theorem (θ l : Real) (h1 : θ = 2) (h2 : l = 2) :
  let r := l / θ
  (1 / 2) * r^2 * θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_theorem_l3698_369840


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3698_369878

theorem complex_equation_sum (x y : ℝ) : 
  (Complex.mk (x - 1) (y + 1)) * (Complex.mk 2 1) = 0 → x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3698_369878


namespace NUMINAMATH_CALUDE_baker_april_earnings_l3698_369806

def baker_earnings (cake_price cake_sold pie_price pie_sold bread_price bread_sold cookie_price cookie_sold pie_discount tax_rate : ℚ) : ℚ :=
  let cake_revenue := cake_price * cake_sold
  let pie_revenue := pie_price * pie_sold * (1 - pie_discount)
  let bread_revenue := bread_price * bread_sold
  let cookie_revenue := cookie_price * cookie_sold
  let total_revenue := cake_revenue + pie_revenue + bread_revenue + cookie_revenue
  total_revenue * (1 + tax_rate)

theorem baker_april_earnings :
  baker_earnings 12 453 7 126 3.5 95 1.5 320 0.1 0.05 = 7394.42 := by
  sorry

end NUMINAMATH_CALUDE_baker_april_earnings_l3698_369806


namespace NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3698_369887

theorem difference_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check difference_of_roots_quadratic 1 (-9) 18

end NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3698_369887


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l3698_369876

def S : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,4}
def N : Set Nat := {2,4}

theorem complement_intersection_equality : 
  (S \ M) ∩ (S \ N) = {3,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l3698_369876


namespace NUMINAMATH_CALUDE_gcd_of_90_and_405_l3698_369860

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_405_l3698_369860


namespace NUMINAMATH_CALUDE_paperboy_delivery_sequences_l3698_369890

/-- Recurrence relation for the number of valid delivery sequences -/
def D : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 4 => D (n + 3) + D (n + 2) + D (n + 1)

/-- Number of valid delivery sequences ending with a delivery -/
def E (n : ℕ) : ℕ := D (n - 2)

/-- The number of houses on King's Avenue -/
def num_houses : ℕ := 15

theorem paperboy_delivery_sequences :
  E num_houses = 3136 := by sorry

end NUMINAMATH_CALUDE_paperboy_delivery_sequences_l3698_369890


namespace NUMINAMATH_CALUDE_inequality_theorem_l3698_369871

theorem inequality_theorem (a b c : ℝ) : a > -b → c - a < c + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3698_369871


namespace NUMINAMATH_CALUDE_equilateral_triangles_with_squares_l3698_369854

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  sorry

/-- Constructs a square externally on a side of a triangle -/
def construct_external_square (t : Triangle) (side : Fin 3) : Square :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem equilateral_triangles_with_squares
  (ABC BCD : Triangle)
  (ABEF : Square)
  (CDGH : Square)
  (h1 : is_equilateral ABC)
  (h2 : is_equilateral BCD)
  (h3 : ABEF = construct_external_square ABC 0)
  (h4 : CDGH = construct_external_square BCD 1)
  : distance ABEF.C CDGH.C / distance ABC.B ABC.C = 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangles_with_squares_l3698_369854


namespace NUMINAMATH_CALUDE_total_calories_burned_first_week_l3698_369808

def calories_per_hour_walking : ℕ := 300

def calories_per_hour_dancing : ℕ := 2 * calories_per_hour_walking

def calories_per_hour_swimming : ℕ := (3 * calories_per_hour_walking) / 2

def calories_per_hour_cycling : ℕ := calories_per_hour_walking

def dancing_hours_per_week : ℕ := 3 * (2 * 1/2) + 1

def swimming_hours_per_week : ℕ := 2 * 3/2

def cycling_hours_per_week : ℕ := 2

def total_calories_burned : ℕ := 
  calories_per_hour_dancing * dancing_hours_per_week +
  calories_per_hour_swimming * swimming_hours_per_week +
  calories_per_hour_cycling * cycling_hours_per_week

theorem total_calories_burned_first_week : 
  total_calories_burned = 4350 := by sorry

end NUMINAMATH_CALUDE_total_calories_burned_first_week_l3698_369808


namespace NUMINAMATH_CALUDE_no_integer_roots_l3698_369839

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0) (ha_even : Even a) (hb_even : Even b) (hc_odd : Odd c) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3698_369839


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l3698_369873

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x+a)(x-4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four (a : ℝ) :
  IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l3698_369873


namespace NUMINAMATH_CALUDE_det_dilation_matrix_det_dilation_matrix_7_l3698_369851

def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0],
    ![0, scale_factor]]

theorem det_dilation_matrix (scale_factor : ℝ) :
  Matrix.det (dilation_matrix scale_factor) = scale_factor ^ 2 := by
  sorry

theorem det_dilation_matrix_7 :
  Matrix.det (dilation_matrix 7) = 49 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_det_dilation_matrix_7_l3698_369851


namespace NUMINAMATH_CALUDE_max_value_inequality_l3698_369898

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≤ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3698_369898


namespace NUMINAMATH_CALUDE_product_equality_l3698_369819

theorem product_equality : 2.05 * 4.1 = 20.5 * 0.41 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3698_369819


namespace NUMINAMATH_CALUDE_ochos_friends_l3698_369838

theorem ochos_friends (total : ℕ) (boys girls : ℕ) (h1 : boys = girls) (h2 : boys + girls = total) (h3 : boys = 4) : total = 8 := by
  sorry

end NUMINAMATH_CALUDE_ochos_friends_l3698_369838


namespace NUMINAMATH_CALUDE_ribbon_per_gift_l3698_369859

theorem ribbon_per_gift (total_gifts : ℕ) (total_ribbon : ℝ) (remaining_ribbon : ℝ) 
  (h1 : total_gifts = 8)
  (h2 : total_ribbon = 15)
  (h3 : remaining_ribbon = 3) :
  (total_ribbon - remaining_ribbon) / total_gifts = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_per_gift_l3698_369859


namespace NUMINAMATH_CALUDE_parking_lot_cars_l3698_369891

theorem parking_lot_cars (car_wheels : ℕ) (motorcycle_wheels : ℕ) (num_motorcycles : ℕ) (total_wheels : ℕ) :
  car_wheels = 5 →
  motorcycle_wheels = 2 →
  num_motorcycles = 11 →
  total_wheels = 117 →
  ∃ num_cars : ℕ, num_cars * car_wheels + num_motorcycles * motorcycle_wheels = total_wheels ∧ num_cars = 19 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l3698_369891


namespace NUMINAMATH_CALUDE_extremum_point_and_monotonicity_l3698_369857

noncomputable section

variables (x : ℝ) (m : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

def f_derivative (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_point_and_monotonicity :
  (f_derivative 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_derivative x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_derivative x < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_extremum_point_and_monotonicity_l3698_369857


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3698_369895

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -4376 [ZMOD 10] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3698_369895


namespace NUMINAMATH_CALUDE_frank_riding_time_l3698_369835

-- Define the riding times for each person
def dave_time : ℝ := 10

-- Chuck's time is 5 times Dave's time
def chuck_time : ℝ := 5 * dave_time

-- Erica's time is 30% longer than Chuck's time
def erica_time : ℝ := chuck_time * (1 + 0.3)

-- Frank's time is 20% longer than Erica's time
def frank_time : ℝ := erica_time * (1 + 0.2)

-- Theorem to prove
theorem frank_riding_time : frank_time = 78 := by
  sorry

end NUMINAMATH_CALUDE_frank_riding_time_l3698_369835


namespace NUMINAMATH_CALUDE_inequality_with_negative_multiplication_l3698_369810

theorem inequality_with_negative_multiplication 
  (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_negative_multiplication_l3698_369810


namespace NUMINAMATH_CALUDE_periodic_exponential_function_l3698_369844

theorem periodic_exponential_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x ∈ Set.Icc (-1) 1, f x = 2^(x + a)) →
  f 2017 = 8 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_exponential_function_l3698_369844


namespace NUMINAMATH_CALUDE_max_k_value_l3698_369825

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ 3/7 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l3698_369825


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3698_369833

theorem fraction_decomposition :
  ∃ (C D : ℝ),
    (C = -0.1 ∧ D = 7.3) ∧
    ∀ (x : ℝ), x ≠ 2 ∧ 3*x ≠ -4 →
      (7*x - 15) / (3*x^2 + 2*x - 8) = C / (x - 2) + D / (3*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3698_369833


namespace NUMINAMATH_CALUDE_expression_percentage_l3698_369802

theorem expression_percentage (x : ℝ) (h : x > 0) : 
  (x / 50 + x / 25 - x / 10 + x / 5) / x = 16 / 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_l3698_369802


namespace NUMINAMATH_CALUDE_lateral_surface_area_equilateral_prism_l3698_369894

/-- The lateral surface area of a prism with an equilateral triangular base -/
theorem lateral_surface_area_equilateral_prism (a : ℝ) (h : a > 0) :
  let base_side := a
  let base_center_to_vertex := a * Real.sqrt 3 / 3
  let edge_angle := 60 * π / 180
  let edge_length := 2 * base_center_to_vertex / Real.cos edge_angle
  let lateral_perimeter := a + a * Real.sqrt 13 / 2
  lateral_perimeter * edge_length = a^2 * Real.sqrt 3 * (Real.sqrt 13 + 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_equilateral_prism_l3698_369894


namespace NUMINAMATH_CALUDE_abs_function_domain_range_intersection_l3698_369809

def A : Set ℝ := {-1, 0, 1}

def f (x : ℝ) : ℝ := |x|

theorem abs_function_domain_range_intersection :
  (A ∩ (f '' A)) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_abs_function_domain_range_intersection_l3698_369809


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l3698_369883

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 / 3) = 27 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l3698_369883


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l3698_369820

/-- Two similar triangles where one has side lengths 12, 12, and 15, and the other has longest side 30 -/
structure SimilarTriangles where
  /-- Side lengths of the smaller triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Longest side of the larger triangle -/
  longest_side : ℝ
  /-- The smaller triangle is isosceles -/
  h_isosceles : a = b
  /-- The side lengths of the smaller triangle -/
  h_sides : a = 12 ∧ c = 15
  /-- The longest side of the larger triangle -/
  h_longest : longest_side = 30

/-- The perimeter of the larger triangle is 78 -/
theorem larger_triangle_perimeter (t : SimilarTriangles) : 
  (t.longest_side / t.c) * (t.a + t.b + t.c) = 78 := by
  sorry

end NUMINAMATH_CALUDE_larger_triangle_perimeter_l3698_369820


namespace NUMINAMATH_CALUDE_lcm_of_36_and_220_l3698_369852

theorem lcm_of_36_and_220 : Nat.lcm 36 220 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_220_l3698_369852


namespace NUMINAMATH_CALUDE_sum_squares_equals_two_l3698_369869

theorem sum_squares_equals_two (x y z : ℝ) 
  (sum_eq : x + y = 2) 
  (product_eq : x * y = z^2 + 1) : 
  x^2 + y^2 + z^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_equals_two_l3698_369869


namespace NUMINAMATH_CALUDE_kiera_had_one_fruit_cup_l3698_369831

/-- Represents the breakfast items and their costs -/
structure Breakfast where
  muffin_cost : ℕ
  fruit_cup_cost : ℕ
  francis_muffins : ℕ
  francis_fruit_cups : ℕ
  kiera_muffins : ℕ
  total_cost : ℕ

/-- Calculates the number of fruit cups Kiera had -/
def kieras_fruit_cups (b : Breakfast) : ℕ :=
  (b.total_cost - (b.francis_muffins * b.muffin_cost + b.francis_fruit_cups * b.fruit_cup_cost + b.kiera_muffins * b.muffin_cost)) / b.fruit_cup_cost

/-- Theorem stating that Kiera had 1 fruit cup given the problem conditions -/
theorem kiera_had_one_fruit_cup (b : Breakfast) 
  (h1 : b.muffin_cost = 2)
  (h2 : b.fruit_cup_cost = 3)
  (h3 : b.francis_muffins = 2)
  (h4 : b.francis_fruit_cups = 2)
  (h5 : b.kiera_muffins = 2)
  (h6 : b.total_cost = 17) :
  kieras_fruit_cups b = 1 := by
  sorry

#eval kieras_fruit_cups { muffin_cost := 2, fruit_cup_cost := 3, francis_muffins := 2, francis_fruit_cups := 2, kiera_muffins := 2, total_cost := 17 }

end NUMINAMATH_CALUDE_kiera_had_one_fruit_cup_l3698_369831
