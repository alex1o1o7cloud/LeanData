import Mathlib

namespace tom_profit_l3185_318587

/-- Represents the types of properties Tom mows --/
inductive PropertyType
| Small
| Medium
| Large

/-- Calculates the total earnings from lawn mowing --/
def lawnMowingEarnings (smallCount medium_count largeCount : ℕ) : ℕ :=
  12 * smallCount + 15 * medium_count + 20 * largeCount

/-- Calculates the total earnings from side tasks --/
def sideTaskEarnings (taskCount : ℕ) : ℕ :=
  10 * taskCount

/-- Calculates the total expenses --/
def totalExpenses : ℕ := 20 + 10

/-- Calculates the total profit --/
def totalProfit (lawnEarnings sideEarnings : ℕ) : ℕ :=
  lawnEarnings + sideEarnings - totalExpenses

/-- Theorem stating Tom's profit for the given month --/
theorem tom_profit :
  totalProfit (lawnMowingEarnings 2 2 1) (sideTaskEarnings 5) = 94 := by
  sorry

end tom_profit_l3185_318587


namespace smallest_double_when_two_moved_l3185_318597

def ends_with_two (n : ℕ) : Prop := n % 10 = 2

def move_two_to_front (n : ℕ) : ℕ :=
  let s := toString n
  let len := s.length
  if len > 1 then
    let front := s.dropRight 1
    let last := s.takeRight 1
    (last ++ front).toNat!
  else n

theorem smallest_double_when_two_moved : ∃ (n : ℕ),
  ends_with_two n ∧
  move_two_to_front n = 2 * n ∧
  ∀ (m : ℕ), m < n → ¬(ends_with_two m ∧ move_two_to_front m = 2 * m) ∧
  n = 105263157894736842 :=
sorry

end smallest_double_when_two_moved_l3185_318597


namespace candy_count_is_twelve_l3185_318574

/-- The total number of candy pieces Wendy and her brother have -/
def total_candy (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  brother_candy + wendy_boxes * pieces_per_box

/-- Theorem: The total number of candy pieces Wendy and her brother have is 12 -/
theorem candy_count_is_twelve :
  total_candy 6 2 3 = 12 := by
  sorry

end candy_count_is_twelve_l3185_318574


namespace periodic_function_l3185_318541

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end periodic_function_l3185_318541


namespace negation_of_for_all_leq_zero_l3185_318522

theorem negation_of_for_all_leq_zero :
  (¬ ∀ x : ℝ, Real.exp x - 2 * Real.sin x + 4 ≤ 0) ↔
  (∃ x : ℝ, Real.exp x - 2 * Real.sin x + 4 > 0) :=
by sorry

end negation_of_for_all_leq_zero_l3185_318522


namespace euler_family_mean_age_l3185_318569

-- Define the list of ages
def euler_family_ages : List ℕ := [6, 6, 9, 11, 13, 16]

-- Theorem statement
theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  (sum_ages : ℚ) / num_children = 61 / 6 := by sorry

end euler_family_mean_age_l3185_318569


namespace equal_area_line_coeff_sum_l3185_318537

/-- A region formed by eight unit circles packed in the first quadrant --/
def R : Set (ℝ × ℝ) :=
  sorry

/-- A line with slope 3 that divides R into two equal areas --/
def l : Set (ℝ × ℝ) :=
  sorry

/-- The line l expressed in the form ax = by + c --/
def line_equation (a b c : ℕ) : Prop :=
  ∀ x y, (x, y) ∈ l ↔ a * x = b * y + c

/-- The coefficients a, b, and c are positive integers with gcd 1 --/
def coeff_constraints (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.gcd a (Nat.gcd b c) = 1

theorem equal_area_line_coeff_sum :
  ∃ a b c : ℕ,
    line_equation a b c ∧
    coeff_constraints a b c ∧
    a^2 + b^2 + c^2 = 65 :=
sorry

end equal_area_line_coeff_sum_l3185_318537


namespace min_value_reciprocal_sum_l3185_318501

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 4 / y) ≥ 9 := by
  sorry

end min_value_reciprocal_sum_l3185_318501


namespace max_savings_bread_l3185_318595

/-- Represents the pricing structure for raisin bread -/
structure BreadPricing where
  single : Rat
  seven : Rat
  dozen : Rat

/-- Calculates the cost of buying bread given a pricing structure and quantities -/
def calculateCost (pricing : BreadPricing) (singles sevens dozens : Nat) : Rat :=
  pricing.single * singles + pricing.seven * sevens + pricing.dozen * dozens

/-- Theorem stating the maximum amount that can be saved -/
theorem max_savings_bread (pricing : BreadPricing) 
  (h1 : pricing.single = 3/10)
  (h2 : pricing.seven = 1)
  (h3 : pricing.dozen = 9/5)
  (budget : Rat)
  (h4 : budget = 10) :
  ∃ (singles sevens dozens : Nat),
    let total_pieces := singles + 7 * sevens + 12 * dozens
    let cost := calculateCost pricing singles sevens dozens
    total_pieces ≥ 60 ∧ cost ≤ budget ∧ budget - cost = 6/5 ∧
    ∀ (s s' d' : Nat), 
      let total_pieces' := s + 7 * s' + 12 * d'
      let cost' := calculateCost pricing s s' d'
      total_pieces' ≥ 60 ∧ cost' ≤ budget → budget - cost' ≤ 6/5 :=
by sorry

end max_savings_bread_l3185_318595


namespace function_evaluation_l3185_318533

theorem function_evaluation (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 1) = x^2 + 1) : 
  f (-1) = 1 := by
sorry

end function_evaluation_l3185_318533


namespace intersection_dot_product_l3185_318532

/-- Given a line ax + by + c = 0 intersecting a circle x^2 + y^2 = 4 at points A and B,
    prove that the dot product of OA and OB is -2 when c^2 = a^2 + b^2 -/
theorem intersection_dot_product
  (a b c : ℝ) 
  (A B : ℝ × ℝ)
  (h1 : ∀ (x y : ℝ), a * x + b * y + c = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B)
  (h2 : c^2 = a^2 + b^2) :
  A.1 * B.1 + A.2 * B.2 = -2 :=
by sorry

end intersection_dot_product_l3185_318532


namespace plane_equation_transformation_l3185_318586

theorem plane_equation_transformation (A B C D : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) :
  ∃ p q r : ℝ, 
    (∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔ x / p + y / q + z / r = 1) ∧
    p = -D / A ∧ q = -D / B ∧ r = -D / C :=
by sorry

end plane_equation_transformation_l3185_318586


namespace random_selection_probability_l3185_318566

theorem random_selection_probability (m : ℝ) : 
  (m > 0) → 
  (2 * m) / (4 - (-2)) = 1 / 3 → 
  m = 1 := by
sorry

end random_selection_probability_l3185_318566


namespace complex_reciprocal_sum_l3185_318512

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end complex_reciprocal_sum_l3185_318512


namespace number_of_students_selected_l3185_318515

/-- Given a class with boys and girls, prove that the number of students selected is 3 -/
theorem number_of_students_selected
  (num_boys : ℕ)
  (num_girls : ℕ)
  (num_ways : ℕ)
  (h_boys : num_boys = 13)
  (h_girls : num_girls = 10)
  (h_ways : num_ways = 780)
  (h_combination : num_ways = (num_girls.choose 1) * (num_boys.choose 2)) :
  3 = 1 + 2 := by
  sorry

#check number_of_students_selected

end number_of_students_selected_l3185_318515


namespace product_of_max_min_a_l3185_318589

theorem product_of_max_min_a (a b c : ℝ) 
  (sum_eq : a + b + c = 15) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 100) : 
  let f := fun x : ℝ => (5 + (5 * Real.sqrt 6) / 3) * (5 - (5 * Real.sqrt 6) / 3)
  f a = 25 / 3 := by
sorry

end product_of_max_min_a_l3185_318589


namespace equal_chord_length_l3185_318508

/-- Given a circle C and two lines l1 and l2, prove that they intercept chords of equal length on C -/
theorem equal_chord_length (r d : ℝ) (h : r > 0) :
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let l1 := {p : ℝ × ℝ | 2 * p.1 + 3 * p.2 + 1 = 0}
  let l2 := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}
  let chord_length (l : Set (ℝ × ℝ)) := 
    Real.sqrt (4 * r^2 - 4 * (1 / (2^2 + 3^2)))
  chord_length l1 = d → chord_length l2 = d :=
by sorry

end equal_chord_length_l3185_318508


namespace min_sum_inequality_l3185_318580

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) := by
  sorry

end min_sum_inequality_l3185_318580


namespace subset_implies_a_values_l3185_318500

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end subset_implies_a_values_l3185_318500


namespace cubic_inequality_l3185_318571

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 30*x > 0 ↔ (0 < x ∧ x < 5) ∨ (x > 6) := by
  sorry

end cubic_inequality_l3185_318571


namespace negation_of_universal_proposition_l3185_318550

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x > 2) ↔ (∃ x : ℝ, x ≥ 1 ∧ x ≤ 2) := by sorry

end negation_of_universal_proposition_l3185_318550


namespace albert_number_puzzle_l3185_318519

theorem albert_number_puzzle (n : ℕ) : 
  (1 : ℚ) / n + (1 : ℚ) / 2 = (1 : ℚ) / 3 + (2 : ℚ) / (n + 1) ↔ n = 2 ∨ n = 3 := by
  sorry

end albert_number_puzzle_l3185_318519


namespace midpoint_tetrahedron_volume_ratio_l3185_318530

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices, just that it's a regular tetrahedron
  is_regular : Bool

/-- The tetrahedron formed by connecting the midpoints of the edges of a regular tetrahedron -/
def midpoint_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  { is_regular := true }  -- The midpoint tetrahedron is also regular

/-- The volume of a tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ :=
  sorry  -- We don't need to define this explicitly for the theorem

/-- 
  The ratio of the volume of the midpoint tetrahedron to the volume of the original tetrahedron
  is 1/8
-/
theorem midpoint_tetrahedron_volume_ratio (t : RegularTetrahedron) :
  volume (midpoint_tetrahedron t) / volume t = 1 / 8 :=
by sorry

end midpoint_tetrahedron_volume_ratio_l3185_318530


namespace product_xyz_equals_2898_l3185_318518

theorem product_xyz_equals_2898 (x y z : ℝ) 
  (eq1 : -3*x + 4*y - z = 28)
  (eq2 : 3*x - 2*y + z = 8)
  (eq3 : x + y - z = 2) :
  x * y * z = 2898 := by sorry

end product_xyz_equals_2898_l3185_318518


namespace five_balls_four_boxes_l3185_318506

/-- The number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end five_balls_four_boxes_l3185_318506


namespace shaded_cubes_count_l3185_318517

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure Cube4x4x4 where
  total_cubes : Nat
  face_size : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in a 4x4x4 cube -/
def count_shaded_cubes (cube : Cube4x4x4) : Nat :=
  sorry

/-- Theorem stating that 24 cubes are shaded on at least one face -/
theorem shaded_cubes_count (cube : Cube4x4x4) 
  (h1 : cube.total_cubes = 64)
  (h2 : cube.face_size = 4)
  (h3 : cube.shaded_per_face = 8) : 
  count_shaded_cubes cube = 24 := by
  sorry

end shaded_cubes_count_l3185_318517


namespace rook_placement_theorem_l3185_318528

theorem rook_placement_theorem (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  ∃ (coloring : Fin n → Fin n → Fin (n^2/2))
    (rook_positions : Fin n → Fin n × Fin n),
    (∀ i j : Fin n, (∃! k : Fin n, coloring i k = coloring j k) ∨ i = j) ∧
    (∀ i j : Fin n, i ≠ j →
      (rook_positions i).1 ≠ (rook_positions j).1 ∧
      (rook_positions i).2 ≠ (rook_positions j).2) ∧
    (∀ i j : Fin n, i ≠ j →
      coloring (rook_positions i).1 (rook_positions i).2 ≠
      coloring (rook_positions j).1 (rook_positions j).2) :=
by sorry

end rook_placement_theorem_l3185_318528


namespace julia_tag_game_l3185_318577

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end julia_tag_game_l3185_318577


namespace fourth_member_income_l3185_318531

/-- Proves that in a family of 4 members with a given average income and known incomes of 3 members, the income of the fourth member is as calculated. -/
theorem fourth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 : ℕ)
  (h1 : family_size = 4)
  (h2 : average_income = 10000)
  (h3 : income1 = 15000)
  (h4 : income2 = 6000)
  (h5 : income3 = 11000) :
  (family_size * average_income) - (income1 + income2 + income3) = 8000 := by
  sorry

#eval (4 * 10000) - (15000 + 6000 + 11000)

end fourth_member_income_l3185_318531


namespace construction_equation_correct_l3185_318505

/-- Represents a construction project with a work stoppage -/
structure ConstructionProject where
  totalLength : ℝ
  originalDailyRate : ℝ
  workStoppageDays : ℝ
  increasedDailyRate : ℝ

/-- The equation correctly represents the construction project situation -/
theorem construction_equation_correct (project : ConstructionProject) 
  (h1 : project.totalLength = 2000)
  (h2 : project.workStoppageDays = 3)
  (h3 : project.increasedDailyRate = project.originalDailyRate + 40) :
  project.totalLength / project.originalDailyRate - 
  project.totalLength / project.increasedDailyRate = 
  project.workStoppageDays := by
sorry

end construction_equation_correct_l3185_318505


namespace smallest_base_for_80_l3185_318535

theorem smallest_base_for_80 :
  ∀ b : ℕ, b ≥ 5 → b^2 ≤ 80 ∧ 80 < b^3 →
  ∀ c : ℕ, c < 5 → ¬(c^2 ≤ 80 ∧ 80 < c^3) :=
by sorry

end smallest_base_for_80_l3185_318535


namespace numerals_with_prime_first_digit_l3185_318561

/-- The set of prime digits less than 10 -/
def primedigits : Finset ℕ := {2, 3, 5, 7}

/-- The number of numerals with prime first digit -/
def num_numerals : ℕ := 400

/-- The number of digits in the numerals -/
def num_digits : ℕ := 3

theorem numerals_with_prime_first_digit :
  (primedigits.card : ℝ) * (10 ^ (num_digits - 1)) = num_numerals := by sorry

end numerals_with_prime_first_digit_l3185_318561


namespace man_speed_in_still_water_l3185_318584

/-- The speed of the man in still water -/
def man_speed : ℝ := 10

/-- The speed of the stream -/
noncomputable def stream_speed : ℝ := sorry

/-- The downstream distance -/
def downstream_distance : ℝ := 28

/-- The upstream distance -/
def upstream_distance : ℝ := 12

/-- The time taken for both upstream and downstream journeys -/
def journey_time : ℝ := 2

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 10 := by
sorry

end man_speed_in_still_water_l3185_318584


namespace cubic_decreasing_implies_m_negative_l3185_318594

def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

theorem cubic_decreasing_implies_m_negative :
  ∀ m : ℝ, (∀ x y : ℝ, x < y → f m x > f m y) → m < 0 :=
by sorry

end cubic_decreasing_implies_m_negative_l3185_318594


namespace saree_price_calculation_l3185_318538

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.2) * (1 - 0.15) = 231.2 → P = 340 := by
  sorry

end saree_price_calculation_l3185_318538


namespace blue_candy_count_l3185_318578

theorem blue_candy_count (total red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end blue_candy_count_l3185_318578


namespace not_right_angled_triangle_l3185_318590

theorem not_right_angled_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

#check not_right_angled_triangle

end not_right_angled_triangle_l3185_318590


namespace subset_arithmetic_result_l3185_318583

theorem subset_arithmetic_result (M : Finset ℕ) 
  (h_card : M.card = 13)
  (h_bounds : ∀ m ∈ M, 100 ≤ m ∧ m ≤ 999) :
  ∃ S : Finset ℕ, S ⊆ M ∧ 
  ∃ a b c d e f : ℕ, 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    3 < (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e ∧
    (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e < 4 :=
sorry

end subset_arithmetic_result_l3185_318583


namespace cubic_equation_solution_l3185_318592

theorem cubic_equation_solution : 
  ∃ x : ℝ, x^3 + 2*(x+1)^3 + (x+2)^3 = (x+4)^3 ∧ x = 3 :=
by
  sorry

end cubic_equation_solution_l3185_318592


namespace simplify_expression_l3185_318553

theorem simplify_expression : 
  1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 - 2 * Real.sqrt 5 := by
  sorry

end simplify_expression_l3185_318553


namespace second_class_average_mark_l3185_318526

theorem second_class_average_mark (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg_total : ℝ) 
  (h1 : students1 = 22)
  (h2 : students2 = 28)
  (h3 : avg1 = 40)
  (h4 : avg_total = 51.2) :
  (avg_total * (students1 + students2) - avg1 * students1) / students2 = 60 := by
  sorry

end second_class_average_mark_l3185_318526


namespace cube_side_length_proof_l3185_318546

/-- The surface area of a cube in square centimeters -/
def surface_area : ℝ := 864

/-- The length of one side of the cube in centimeters -/
def side_length : ℝ := 12

/-- Theorem: For a cube with a surface area of 864 cm², the length of one side is 12 cm -/
theorem cube_side_length_proof :
  6 * side_length ^ 2 = surface_area := by sorry

end cube_side_length_proof_l3185_318546


namespace panda_increase_l3185_318579

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The ratio of cheetahs to pandas is 1:3 -/
def valid_ratio (pop : ZooPopulation) : Prop :=
  3 * pop.cheetahs = pop.pandas

theorem panda_increase (old_pop new_pop : ZooPopulation) :
  valid_ratio old_pop →
  valid_ratio new_pop →
  new_pop.cheetahs = old_pop.cheetahs + 2 →
  new_pop.pandas = old_pop.pandas + 6 := by
  sorry

end panda_increase_l3185_318579


namespace nearest_integer_to_x_minus_y_is_zero_l3185_318527

theorem nearest_integer_to_x_minus_y_is_zero
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : 2 * abs x + y = 5)
  (h2 : abs x * y + x^2 = 0) :
  round (x - y) = 0 :=
sorry

end nearest_integer_to_x_minus_y_is_zero_l3185_318527


namespace expression_value_l3185_318525

theorem expression_value (x y : ℝ) (h : x - y = 1) : 3*x - 3*y + 1 = 4 := by
  sorry

end expression_value_l3185_318525


namespace square_root_problem_l3185_318570

theorem square_root_problem (x y : ℝ) (h : Real.sqrt (2 * x - 16) + |x - 2 * y + 2| = 0) :
  Real.sqrt (x - 4 / 5 * y) = 2 ∨ Real.sqrt (x - 4 / 5 * y) = -2 := by
  sorry

end square_root_problem_l3185_318570


namespace arithmetic_sequence_problem_l3185_318502

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 4 and a₄ = 2, a₆ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 6 = 0 := by
sorry

end arithmetic_sequence_problem_l3185_318502


namespace linear_equation_solution_l3185_318510

theorem linear_equation_solution (x m : ℝ) : 
  4 * x + 2 * m = 14 → x = 2 → m = 3 := by
  sorry

end linear_equation_solution_l3185_318510


namespace angle_function_equality_l3185_318572

/-- Given an angle α in the third quadrant, if cos(α - 3π/2) = 1/5, then
    (sin(α - π/2) * cos(3π/2 + α) * tan(π - α)) / (tan(-α - π) * sin(-α - π)) = 2√6/5 -/
theorem angle_function_equality (α : Real) 
    (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
    (h2 : Real.cos (α - 3*π/2) = 1/5) :
    (Real.sin (α - π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)) / 
    (Real.tan (-α - π) * Real.sin (-α - π)) = 2 * Real.sqrt 6 / 5 := by
  sorry

end angle_function_equality_l3185_318572


namespace cubic_polynomial_value_l3185_318514

/-- A cubic polynomial function. -/
def CubicPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating that a cubic polynomial with given properties has f(1) = -23. -/
theorem cubic_polynomial_value (f : ℝ → ℝ) 
  (hcubic : CubicPolynomial f)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 := by
  sorry

end cubic_polynomial_value_l3185_318514


namespace solution_set_inequality_l3185_318509

theorem solution_set_inequality (x : ℝ) : 
  x * |x + 2| < 0 ↔ x < -2 ∨ (-2 < x ∧ x < 0) :=
by sorry

end solution_set_inequality_l3185_318509


namespace cylinder_height_l3185_318598

/-- For a cylinder with base radius 3, if its lateral surface area is 1/2 of its total surface area, then its height is 3 -/
theorem cylinder_height (h : ℝ) (h_pos : h > 0) : 
  (2 * π * 3 * h) = (1/2) * (2 * π * 3 * h + 2 * π * 3^2) → h = 3 := by
sorry

end cylinder_height_l3185_318598


namespace inverse_function_property_l3185_318543

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 3 = 1 → f 1 = 3 := by
  sorry

end inverse_function_property_l3185_318543


namespace remainder_problem_l3185_318513

theorem remainder_problem : (2^300 + 300) % (2^150 + 2^75 + 1) = 298 := by
  sorry

end remainder_problem_l3185_318513


namespace trigonometric_equation_solution_l3185_318591

theorem trigonometric_equation_solution (x : ℝ) :
  (8.4743 * Real.tan (2 * x) - 4 * Real.tan (3 * x) = Real.tan (3 * x)^2 * Real.tan (2 * x)) ↔
  (∃ k : ℤ, x = k * Real.pi ∨ x = Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi ∨ 
   x = -Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi) :=
by sorry

end trigonometric_equation_solution_l3185_318591


namespace simple_pairs_l3185_318504

theorem simple_pairs (n : ℕ) (h : n > 3) :
  ∃ (p₁ p₂ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Odd p₁ ∧ Odd p₂ ∧ (p₂ ∣ (2 * n - p₁)) :=
sorry

end simple_pairs_l3185_318504


namespace simplify_and_sum_exponents_l3185_318599

theorem simplify_and_sum_exponents 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ), 
    (k > 0) ∧ 
    (k^3 = a^2 * b^4 * c^2) ∧ 
    ((72 * a^5 * b^7 * c^14)^(1/3) = 2 * 3^(2/3) * a * b * c^4 * k) ∧
    (1 + 1 + 4 = 6) := by
  sorry

end simplify_and_sum_exponents_l3185_318599


namespace jerry_birthday_mean_l3185_318596

def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9
def friend_gift1 : ℝ := 22
def friend_gift2 : ℝ := 23
def friend_gift3 : ℝ := 22
def friend_gift4 : ℝ := 22
def sister_gift : ℝ := 7

def total_amount : ℝ := aunt_gift + uncle_gift + friend_gift1 + friend_gift2 + friend_gift3 + friend_gift4 + sister_gift
def number_of_gifts : ℕ := 7

theorem jerry_birthday_mean :
  total_amount / number_of_gifts = 16.29 := by sorry

end jerry_birthday_mean_l3185_318596


namespace critical_point_iff_a_in_range_l3185_318588

/-- The function f(x) = x³ - ax² + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + a*x + 3

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + a

/-- A critical point of f exists if and only if f'(x) = 0 has real solutions -/
def has_critical_point (a : ℝ) : Prop := ∃ x : ℝ, f' a x = 0

/-- The main theorem: f(x) has a critical point iff a ∈ (-∞, 0) ∪ (3, +∞) -/
theorem critical_point_iff_a_in_range (a : ℝ) :
  has_critical_point a ↔ a < 0 ∨ a > 3 := by sorry

end critical_point_iff_a_in_range_l3185_318588


namespace expression_value_l3185_318551

theorem expression_value (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end expression_value_l3185_318551


namespace one_dollar_bills_count_l3185_318511

/-- Represents the wallet contents -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def satisfies_conditions (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 55 ∧
  w.ones * 1 + w.twos * 2 + w.fives * 5 = 126

/-- The theorem stating the number of one-dollar bills -/
theorem one_dollar_bills_count :
  ∃ (w : Wallet), satisfies_conditions w ∧ w.ones = 18 := by
  sorry

end one_dollar_bills_count_l3185_318511


namespace range_of_m_l3185_318516

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 1/3

def q (m : ℝ) : Prop := ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 0 < m ∧ m < 15

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1/3 ≤ m ∧ m < 15 :=
by sorry

end range_of_m_l3185_318516


namespace hyperbola_eccentricity_l3185_318520

-- Define the parabola and hyperbola
def parabola (b : ℝ) (x y : ℝ) : Prop := x^2 = -6*b*y
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the points
def point_O : ℝ × ℝ := (0, 0)
def point_A (a b : ℝ) : ℝ × ℝ := (a, 0)

-- Define the angle equality
def angle_equality (O A B C : ℝ × ℝ) : Prop := 
  (C.2 - O.2) / (C.1 - O.1) = (C.2 - B.2) / (C.1 - B.1)

-- Main theorem
theorem hyperbola_eccentricity (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hB : parabola b (-a*Real.sqrt 13/2) (3*b/2))
  (hC : parabola b (a*Real.sqrt 13/2) (3*b/2))
  (hBC : hyperbola a b (-a*Real.sqrt 13/2) (3*b/2) ∧ 
         hyperbola a b (a*Real.sqrt 13/2) (3*b/2))
  (hAOC : angle_equality point_O (point_A a b) 
    (-a*Real.sqrt 13/2, 3*b/2) (a*Real.sqrt 13/2, 3*b/2)) :
  Real.sqrt (1 + b^2/a^2) = 4*Real.sqrt 3/3 := by
sorry

end hyperbola_eccentricity_l3185_318520


namespace arithmetic_mean_problem_l3185_318559

theorem arithmetic_mean_problem (a : ℝ) : 
  (1 + a) / 2 = 2 → a = 3 := by
  sorry

end arithmetic_mean_problem_l3185_318559


namespace linear_equation_and_absolute_value_l3185_318534

theorem linear_equation_and_absolute_value (m a : ℝ) :
  (∀ x, (m^2 - 9) * x^2 - (m - 3) * x + 6 = 0 → (m^2 - 9 = 0 ∧ m - 3 ≠ 0)) →
  |a| ≤ |m| →
  |a + m| + |a - m| = 6 := by
sorry

end linear_equation_and_absolute_value_l3185_318534


namespace hexagon_perimeter_l3185_318556

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by
  sorry

end hexagon_perimeter_l3185_318556


namespace max_salary_is_220000_l3185_318564

/-- Represents a basketball team with salary constraints -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  salary_cap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player -/
def max_highest_salary (team : BasketballTeam) : ℕ :=
  team.salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for the highest-paid player -/
theorem max_salary_is_220000 (team : BasketballTeam) 
  (h1 : team.num_players = 15)
  (h2 : team.min_salary = 20000)
  (h3 : team.salary_cap = 500000) :
  max_highest_salary team = 220000 := by
  sorry

#eval max_highest_salary { num_players := 15, min_salary := 20000, salary_cap := 500000 }

end max_salary_is_220000_l3185_318564


namespace max_taxiing_time_l3185_318524

/-- The function representing the distance traveled by the plane after landing -/
def y (t : ℝ) : ℝ := 60 * t - 2 * t^2

/-- The maximum time the plane uses for taxiing -/
def s : ℝ := 15

theorem max_taxiing_time :
  ∀ t : ℝ, y t ≤ y s :=
by sorry

end max_taxiing_time_l3185_318524


namespace major_axis_length_is_6_l3185_318552

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The ellipse is tangent to the line x = 4 -/
  tangent_x_4 : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinates of the foci -/
  foci_y1 : ℝ
  foci_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 6 -/
theorem major_axis_length_is_6 (e : Ellipse) 
  (h1 : e.tangent_y_axis = true) 
  (h2 : e.tangent_x_4 = true)
  (h3 : e.foci_x = 3)
  (h4 : e.foci_y1 = 1 + Real.sqrt 3)
  (h5 : e.foci_y2 = 1 - Real.sqrt 3) : 
  majorAxisLength e = 6 := by sorry

end major_axis_length_is_6_l3185_318552


namespace enchanted_creatures_gala_handshakes_l3185_318557

/-- The number of handshakes at the Enchanted Creatures Gala -/
theorem enchanted_creatures_gala_handshakes : 
  let num_goblins : ℕ := 30
  let num_trolls : ℕ := 20
  let goblin_handshakes := num_goblins * (num_goblins - 1) / 2
  let goblin_troll_handshakes := num_goblins * num_trolls
  goblin_handshakes + goblin_troll_handshakes = 1035 := by
  sorry

#check enchanted_creatures_gala_handshakes

end enchanted_creatures_gala_handshakes_l3185_318557


namespace unique_number_property_l3185_318568

theorem unique_number_property : ∃! (a : ℕ), a > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (a^6 - 1) → (p ∣ (a^3 - 1) ∨ p ∣ (a^2 - 1))) →
  a = 2 :=
by sorry

end unique_number_property_l3185_318568


namespace matrix_power_difference_l3185_318560

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference : 
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end matrix_power_difference_l3185_318560


namespace percentage_loss_is_twenty_percent_l3185_318581

/-- Calculates the percentage loss given the selling conditions --/
def calculate_percentage_loss (initial_articles : ℕ) (initial_price : ℚ) (initial_gain_percent : ℚ) 
  (final_articles : ℚ) (final_price : ℚ) : ℚ :=
  let initial_cost := initial_price / (1 + initial_gain_percent / 100)
  let cost_per_article := initial_cost / initial_articles
  let final_cost := cost_per_article * final_articles
  let loss := final_cost - final_price
  (loss / final_cost) * 100

/-- The percentage loss is 20% given the specified conditions --/
theorem percentage_loss_is_twenty_percent :
  calculate_percentage_loss 20 60 20 20 40 = 20 := by
  sorry

end percentage_loss_is_twenty_percent_l3185_318581


namespace five_student_committees_from_eight_l3185_318565

theorem five_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end five_student_committees_from_eight_l3185_318565


namespace g_of_three_value_l3185_318536

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all x ≠ 0 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The main theorem stating that g(3) = 36.333/7 -/
theorem g_of_three_value : g 3 = 36.333 / 7 := by
  sorry

end g_of_three_value_l3185_318536


namespace square_side_length_l3185_318545

theorem square_side_length (x : ℝ) (h : x > 0) : x^2 = 2 * (4 * x) → x = 8 := by
  sorry

end square_side_length_l3185_318545


namespace final_cat_count_l3185_318539

def initial_siamese : ℝ := 13.5
def initial_house : ℝ := 5.25
def cats_added : ℝ := 10.75
def cats_discounted : ℝ := 0.5

theorem final_cat_count :
  initial_siamese + initial_house + cats_added - cats_discounted = 29 := by
  sorry

end final_cat_count_l3185_318539


namespace investment_bankers_count_l3185_318563

/-- Proves that the number of investment bankers is 4 given the problem conditions -/
theorem investment_bankers_count : 
  ∀ (total_bill : ℝ) (avg_cost : ℝ) (num_clients : ℕ),
  total_bill = 756 →
  avg_cost = 70 →
  num_clients = 5 →
  ∃ (num_bankers : ℕ),
    num_bankers = 4 ∧
    total_bill = (avg_cost * (num_bankers + num_clients : ℝ)) * 1.2 :=
by sorry

end investment_bankers_count_l3185_318563


namespace feifei_arrival_time_l3185_318554

/-- Represents the speed of an entity -/
structure Speed :=
  (value : ℝ)

/-- Represents a distance -/
structure Distance :=
  (value : ℝ)

/-- Represents a time duration in minutes -/
structure Duration :=
  (minutes : ℝ)

/-- Represents the scenario of Feifei walking to school -/
structure WalkToSchool :=
  (feifei_speed : Speed)
  (dog_speed : Speed)
  (first_catchup : Distance)
  (second_catchup : Distance)
  (total_distance : Distance)
  (dog_start_delay : Duration)

/-- The theorem stating that Feifei arrives at school 18 minutes after starting -/
theorem feifei_arrival_time (scenario : WalkToSchool) 
  (h1 : scenario.dog_speed.value = 3 * scenario.feifei_speed.value)
  (h2 : scenario.first_catchup.value = 200)
  (h3 : scenario.second_catchup.value = 400)
  (h4 : scenario.total_distance.value = 800)
  (h5 : scenario.dog_start_delay.minutes = 3) :
  ∃ (arrival_time : Duration), arrival_time.minutes = 18 :=
sorry

end feifei_arrival_time_l3185_318554


namespace factorization_equality_l3185_318529

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x - 2*y^2 + 4*y - x*y = (x - 2*y)*(x + y - 2) := by
sorry

end factorization_equality_l3185_318529


namespace circular_motion_angle_l3185_318503

theorem circular_motion_angle (θ : Real) : 
  (0 < θ) ∧ (θ < π) ∧                        -- 0 < θ < π
  (π < 2*θ) ∧ (2*θ < 3*π/2) ∧                -- Reaches third quadrant in 2 minutes
  (∃ (n : ℤ), 14*θ = n * (2*π)) →            -- Returns to original position in 14 minutes
  (θ = 4*π/7) ∨ (θ = 5*π/7) := by
sorry

end circular_motion_angle_l3185_318503


namespace polynomial_simplification_l3185_318555

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + 5 * x^2 + 16) - (x^6 + 4 * x^5 - 2 * x^3 + 3 * x^2 + 18) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 + 2 * x^2 - 2 :=
by sorry

end polynomial_simplification_l3185_318555


namespace problem_solution_l3185_318542

theorem problem_solution (a b c : ℝ) 
  (eq : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h1 : b = 15)
  (h2 : c = 5)
  (h3 : 2 = Real.sqrt ((a + 2) * (15 + 3)) / (5 + 1)) :
  a = 6 := by sorry

end problem_solution_l3185_318542


namespace triangle_area_theorem_l3185_318585

-- Define a triangle type
structure Triangle where
  base : ℝ
  median1 : ℝ
  median2 : ℝ

-- Define the area function
def triangleArea (t : Triangle) : ℝ :=
  sorry  -- The actual calculation would go here

-- Theorem statement
theorem triangle_area_theorem (t : Triangle) 
  (h1 : t.base = 20)
  (h2 : t.median1 = 18)
  (h3 : t.median2 = 24) : 
  triangleArea t = 288 := by
  sorry


end triangle_area_theorem_l3185_318585


namespace cos_arcsin_eight_seventeenths_l3185_318567

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by sorry

end cos_arcsin_eight_seventeenths_l3185_318567


namespace polygon_interior_angles_sum_l3185_318549

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 2340) → (180 * ((n - 3) - 2) = 1800) := by sorry

end polygon_interior_angles_sum_l3185_318549


namespace sum_of_specific_values_l3185_318544

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem sum_of_specific_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f_1 : f 1 = 2014) :
  f 2013 + f 2014 + f 2015 = 0 := by
  sorry

end sum_of_specific_values_l3185_318544


namespace taxi_seating_arrangements_l3185_318523

theorem taxi_seating_arrangements :
  let n : ℕ := 6  -- total number of people
  let m : ℕ := 4  -- maximum capacity of each taxi
  let k : ℕ := 2  -- number of taxis
  Nat.choose n m * 2 + (Nat.choose n (n / k)) = 50 :=
by sorry

end taxi_seating_arrangements_l3185_318523


namespace circle_extrema_l3185_318593

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - 3)^2 + (y₁ - 3)^2 = 6 ∧ 
    (x₂ - 3)^2 + (y₂ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → y' / x' ≤ y₁ / x₁ ∧ y' / x' ≥ y₂ / x₂) ∧
    y₁ / x₁ = 3 + 2 * Real.sqrt 2 ∧
    y₂ / x₂ = 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₃ y₃ x₄ y₄ : ℝ),
    (x₃ - 3)^2 + (y₃ - 3)^2 = 6 ∧
    (x₄ - 3)^2 + (y₄ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → 
      Real.sqrt ((x' - 2)^2 + y'^2) ≤ Real.sqrt ((x₃ - 2)^2 + y₃^2) ∧
      Real.sqrt ((x' - 2)^2 + y'^2) ≥ Real.sqrt ((x₄ - 2)^2 + y₄^2)) ∧
    Real.sqrt ((x₃ - 2)^2 + y₃^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    Real.sqrt ((x₄ - 2)^2 + y₄^2) = Real.sqrt 10 - Real.sqrt 6) := by
  sorry

end circle_extrema_l3185_318593


namespace point_on_exponential_graph_l3185_318562

theorem point_on_exponential_graph (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1)
  f 1 = 1 := by sorry

end point_on_exponential_graph_l3185_318562


namespace arithmetic_sequence_solution_l3185_318573

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d < 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 2 * a 4 = 12
  h4 : a 2 + a 4 = 8

/-- The theorem stating the existence of a unique solution and sum of first 10 terms -/
theorem arithmetic_sequence_solution (seq : ArithmeticSequence) :
  ∃! (a₁ : ℝ), 
    (seq.a 1 = a₁) ∧
    (∃! (d : ℝ), d = seq.d) ∧
    (∃ (S₁₀ : ℝ), S₁₀ = (10 * seq.a 1) + (10 * 9 / 2 * seq.d)) :=
sorry

end arithmetic_sequence_solution_l3185_318573


namespace polygon_sides_l3185_318548

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (180 * (n - 2)) = 2 / 9 → n = 11 := by
  sorry

end polygon_sides_l3185_318548


namespace circles_intersect_l3185_318507

/-- The circles x^2 + y^2 + 4x - 4y - 8 = 0 and x^2 + y^2 - 2x + 4y + 1 = 0 intersect. -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 + 4*x - 4*y - 8 = 0) ∧ (x^2 + y^2 - 2*x + 4*y + 1 = 0) := by
  sorry


end circles_intersect_l3185_318507


namespace library_book_count_l3185_318521

theorem library_book_count : ∃ (initial_books : ℕ), 
  initial_books = 1750 ∧ 
  initial_books + 140 = (27 * initial_books) / 25 := by
sorry

end library_book_count_l3185_318521


namespace game_properties_l3185_318582

/-- Represents the "What? Where? When?" game --/
structure Game where
  num_envelopes : Nat
  points_to_win : Nat
  num_games : Nat

/-- Calculates the expected number of points for one team in multiple games --/
def expectedPoints (g : Game) : ℝ :=
  sorry

/-- Calculates the probability of a specific envelope being chosen --/
def envelopeProbability (g : Game) : ℝ :=
  sorry

/-- Theorem stating the expected points and envelope probability for the given game --/
theorem game_properties :
  let g : Game := { num_envelopes := 13, points_to_win := 6, num_games := 100 }
  (expectedPoints g = 465) ∧ (envelopeProbability g = 12 / 13) := by
  sorry

end game_properties_l3185_318582


namespace total_green_peaches_l3185_318558

/-- Represents a basket of peaches -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- Proves that the total number of green peaches is 9 given the conditions -/
theorem total_green_peaches
  (b1 b2 b3 : Basket)
  (h1 : b1.red = 4)
  (h2 : b2.red = 4)
  (h3 : b3.red = 3)
  (h_total : b1.red + b1.green + b2.red + b2.green + b3.red + b3.green = 20) :
  b1.green + b2.green + b3.green = 9 := by
  sorry

end total_green_peaches_l3185_318558


namespace largest_number_l3185_318547

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.989) 
  (hb : b = 0.998) 
  (hc : c = 0.899) 
  (hd : d = 0.9899) 
  (he : e = 0.8999) : 
  b = max a (max b (max c (max d e))) := by
sorry

end largest_number_l3185_318547


namespace hannah_mug_collection_l3185_318576

theorem hannah_mug_collection (total_mugs : ℕ) (num_colors : ℕ) (yellow_mugs : ℕ) :
  total_mugs = 40 →
  num_colors = 4 →
  yellow_mugs = 12 →
  let red_mugs := yellow_mugs / 2
  let blue_mugs := 3 * red_mugs
  total_mugs = blue_mugs + red_mugs + yellow_mugs + (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) →
  (total_mugs - (blue_mugs + red_mugs + yellow_mugs)) = 4 := by
  sorry

end hannah_mug_collection_l3185_318576


namespace negation_of_proposition_l3185_318540

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > 0 → a * b ≥ 0) ↔ (∃ (a b : ℝ), a > 0 ∧ a * b < 0) := by
  sorry

end negation_of_proposition_l3185_318540


namespace optimal_gcd_l3185_318575

/-- The number of integers to choose from (0 to 81 inclusive) -/
def n : ℕ := 82

/-- The set of numbers to choose from -/
def S : Finset ℕ := Finset.range n

/-- Amy's strategy: A function that takes the current state and returns Amy's choice -/
def amy_strategy : S → S → ℕ → ℕ := sorry

/-- Bob's strategy: A function that takes the current state and returns Bob's choice -/
def bob_strategy : S → S → ℕ → ℕ := sorry

/-- The sum of Amy's chosen numbers -/
def A (amy_nums : Finset ℕ) : ℕ := amy_nums.sum id

/-- The sum of Bob's chosen numbers -/
def B (bob_nums : Finset ℕ) : ℕ := bob_nums.sum id

/-- The game result when Amy and Bob play optimally -/
def optimal_play : Finset ℕ × Finset ℕ := sorry

/-- The theorem stating the optimal gcd when Amy and Bob play optimally -/
theorem optimal_gcd :
  let (amy_nums, bob_nums) := optimal_play
  Nat.gcd (A amy_nums) (B bob_nums) = 41 := by sorry

end optimal_gcd_l3185_318575
