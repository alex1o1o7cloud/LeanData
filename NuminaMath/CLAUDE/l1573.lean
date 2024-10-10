import Mathlib

namespace children_distribution_l1573_157362

theorem children_distribution (n : ℕ) : 
  (6 : ℝ) / n - (6 : ℝ) / (n + 2) = (1 : ℝ) / 4 → n + 2 = 8 := by
  sorry

end children_distribution_l1573_157362


namespace set_equality_implies_sum_of_powers_l1573_157391

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2016 = 1 := by
  sorry

end set_equality_implies_sum_of_powers_l1573_157391


namespace roots_modulus_one_preserved_l1573_157385

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) :=
by sorry

end roots_modulus_one_preserved_l1573_157385


namespace find_r_l1573_157322

-- Define the polynomials f and g
def f (r a x : ℝ) : ℝ := (x - (r + 2)) * (x - (r + 6)) * (x - a)
def g (r b x : ℝ) : ℝ := (x - (r + 4)) * (x - (r + 8)) * (x - b)

-- State the theorem
theorem find_r : ∃ (r a b : ℝ), 
  (∀ x, f r a x - g r b x = 2 * r) → r = 48 / 17 := by
  sorry

end find_r_l1573_157322


namespace bobby_blocks_l1573_157304

def total_blocks (initial_blocks : ℕ) (factor : ℕ) : ℕ :=
  initial_blocks + factor * initial_blocks

theorem bobby_blocks : total_blocks 2 3 = 8 := by
  sorry

end bobby_blocks_l1573_157304


namespace trigonometric_ratio_equality_l1573_157327

theorem trigonometric_ratio_equality 
  (a b c α β : ℝ) 
  (eq1 : a * Real.cos α + b * Real.sin α = c)
  (eq2 : a * Real.cos β + b * Real.sin β = c) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    a = k * Real.cos ((α + β) / 2) ∧
    b = k * Real.sin ((α + β) / 2) ∧
    c = k * Real.cos ((α - β) / 2) :=
by sorry

end trigonometric_ratio_equality_l1573_157327


namespace last_i_becomes_w_l1573_157384

/-- Represents a letter in the alphabet --/
def Letter := Fin 26

/-- The encryption shift for the nth occurrence of a letter --/
def shift (n : Nat) : Nat := n^2

/-- The message to be encrypted --/
def message : String := "Mathematics is meticulous"

/-- Count occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : Nat :=
  s.toList.filter (· = c) |>.length

/-- Apply the shift to a letter --/
def applyShift (l : Letter) (s : Nat) : Letter :=
  ⟨(l.val + s) % 26, by sorry⟩

/-- The theorem to be proved --/
theorem last_i_becomes_w :
  let iCount := countOccurrences 'i' message
  let totalShift := (List.range iCount).map shift |>.sum
  let iLetter : Letter := ⟨8, by sorry⟩  -- 'i' is the 9th letter (0-indexed)
  applyShift iLetter totalShift = ⟨22, by sorry⟩  -- 'w' is the 23rd letter (0-indexed)
  := by sorry

end last_i_becomes_w_l1573_157384


namespace james_beat_record_by_296_l1573_157393

/-- Calculates the total points scored by James in a football season -/
def james_total_points (
  touchdowns_per_game : ℕ)
  (touchdown_points : ℕ)
  (games_in_season : ℕ)
  (two_point_conversions : ℕ)
  (field_goals : ℕ)
  (field_goal_points : ℕ)
  (extra_point_attempts : ℕ)
  (bonus_touchdown_sets : ℕ)
  (bonus_touchdowns_per_set : ℕ)
  (bonus_multiplier : ℕ) : ℕ :=
  let regular_touchdown_points := touchdowns_per_game * games_in_season * touchdown_points
  let bonus_touchdown_points := bonus_touchdown_sets * bonus_touchdowns_per_set * touchdown_points * bonus_multiplier
  let two_point_conversion_points := two_point_conversions * 2
  let field_goal_points := field_goals * field_goal_points
  let extra_point_points := extra_point_attempts
  regular_touchdown_points + bonus_touchdown_points + two_point_conversion_points + field_goal_points + extra_point_points

/-- Theorem stating that James beat the old record by 296 points -/
theorem james_beat_record_by_296 :
  james_total_points 4 6 15 6 8 3 20 5 3 2 - 300 = 296 := by
  sorry

end james_beat_record_by_296_l1573_157393


namespace complement_of_union_equals_four_l1573_157398

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_union_equals_four_l1573_157398


namespace cube_surface_area_l1573_157361

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end cube_surface_area_l1573_157361


namespace inequality_proof_l1573_157369

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end inequality_proof_l1573_157369


namespace min_value_x_plus_y_l1573_157352

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2*x + y)⁻¹ + 4*(2*x + 3*y)⁻¹ = 1) :
  x + y ≥ 9/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2*x₀ + y₀)⁻¹ + 4*(2*x₀ + 3*y₀)⁻¹ = 1 ∧ x₀ + y₀ = 9/4 :=
by sorry

end min_value_x_plus_y_l1573_157352


namespace solve_equation_l1573_157333

theorem solve_equation (x y : ℝ) (h : 3 * x^2 - 2 * y = 1) :
  2025 + 2 * y - 3 * x^2 = 2024 := by
  sorry

end solve_equation_l1573_157333


namespace intersection_complement_equality_l1573_157315

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x | 1 ≤ x ∧ x < 2} := by sorry

end intersection_complement_equality_l1573_157315


namespace inequality_equivalence_l1573_157395

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2 + y) / x < (4 - x) / y ↔
  ((x * y > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧
   (x * y < 0 → (x - 2)^2 + (y + 1)^2 > 5)) :=
by sorry

end inequality_equivalence_l1573_157395


namespace prob_other_side_green_l1573_157365

/-- Represents a card with two sides --/
inductive Card
| BlueBoth
| BlueGreen
| GreenBoth

/-- The box of cards --/
def box : Finset Card := sorry

/-- The number of cards in the box --/
def num_cards : ℕ := 8

/-- The number of cards that are blue on both sides --/
def num_blue_both : ℕ := 4

/-- The number of cards that are blue on one side and green on the other --/
def num_blue_green : ℕ := 2

/-- The number of cards that are green on both sides --/
def num_green_both : ℕ := 2

/-- Function to check if a given side of a card is green --/
def is_green (c : Card) (side : Bool) : Bool := sorry

/-- The probability of picking a card and observing a green side --/
def prob_green_side : ℚ := sorry

/-- The probability of both sides being green given that one observed side is green --/
def prob_both_green_given_one_green : ℚ := sorry

theorem prob_other_side_green : 
  prob_both_green_given_one_green = 2/3 := sorry

end prob_other_side_green_l1573_157365


namespace inequality_system_solution_l1573_157317

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end inequality_system_solution_l1573_157317


namespace taut_if_pred_prime_l1573_157380

def is_taut (n : ℕ) : Prop :=
  ∃ (S : Finset (Fin (n^2 - n + 1))),
    S.card = n ∧
    ∀ (a b c d : Fin (n^2 - n + 1)),
      a ∈ S → b ∈ S → c ∈ S → d ∈ S →
      a ≠ b → c ≠ d →
      (a : ℕ) * (d : ℕ) ≠ (b : ℕ) * (c : ℕ)

theorem taut_if_pred_prime (n : ℕ) (h : n ≥ 2) (h_prime : Nat.Prime (n - 1)) :
  is_taut n :=
sorry

end taut_if_pred_prime_l1573_157380


namespace complement_A_complement_B_intersection_A_complement_B_l1573_157300

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5 ∨ x = 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- State the theorems to be proved
theorem complement_A : (Set.univ \ A) = {x | x ≤ -1 ∨ (5 < x ∧ x < 6) ∨ x > 6} := by sorry

theorem complement_B : (Set.univ \ B) = {x | x < 2 ∨ x ≥ 5} := by sorry

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 < x ∧ x < 2 ∨ x = 5 ∨ x = 6} := by sorry

end complement_A_complement_B_intersection_A_complement_B_l1573_157300


namespace orange_apple_weight_equivalence_l1573_157397

/-- Given that 9 oranges weigh the same as 6 apples, prove that 54 oranges
    weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    9 * orange_weight = 6 * apple_weight →
    54 * orange_weight = 36 * apple_weight := by
  sorry

end orange_apple_weight_equivalence_l1573_157397


namespace transportation_theorem_l1573_157394

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  supplies_C : ℝ
  supplies_D : ℝ
  cost_C_to_A : ℝ
  cost_C_to_B : ℝ
  cost_D_to_A : ℝ
  cost_D_to_B : ℝ
  x : ℝ  -- Amount transported from D to B

/-- The total transportation cost as a function of x -/
def total_cost (p : TransportationProblem) : ℝ :=
  p.cost_C_to_A * (200 - (p.supplies_D - p.x)) + 
  p.cost_C_to_B * (300 - p.x) + 
  p.cost_D_to_A * (p.supplies_D - p.x) + 
  p.cost_D_to_B * p.x

theorem transportation_theorem (p : TransportationProblem) 
  (h1 : p.supplies_C = 240)
  (h2 : p.supplies_D = 260)
  (h3 : p.cost_C_to_A = 20)
  (h4 : p.cost_C_to_B = 25)
  (h5 : p.cost_D_to_A = 15)
  (h6 : p.cost_D_to_B = 30)
  (h7 : 60 ≤ p.x ∧ p.x ≤ 260) : 
  (∃ (w : ℝ), w = total_cost p ∧ w = 10 * p.x + 10200) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), 60 ≤ x → x ≤ 260 → 
    (10 - m) * x + 10200 ≥ 10320) ↔ (0 < m ∧ m ≤ 8)) :=
by sorry

end transportation_theorem_l1573_157394


namespace line_circle_intersection_m_values_l1573_157354

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter m in the line equation x - y + m = 0 -/
  m : ℝ
  /-- The line intersects the circle x^2 + y^2 = 4 at two points -/
  intersects : ∃ (A B : ℝ × ℝ), A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧
                                 A.1 - A.2 + m = 0 ∧ B.1 - B.2 + m = 0
  /-- The length of the chord AB is 2√3 -/
  chord_length : ∃ (A B : ℝ × ℝ), (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12

/-- The theorem stating the possible values of m -/
theorem line_circle_intersection_m_values (lci : LineCircleIntersection) :
  lci.m = Real.sqrt 2 ∨ lci.m = -Real.sqrt 2 := by
  sorry

end line_circle_intersection_m_values_l1573_157354


namespace problem_statement_l1573_157331

theorem problem_statement (a : ℝ) (h_pos : a > 0) (h_eq : a^2 / (a^4 - a^2 + 1) = 4/37) :
  a^3 / (a^6 - a^3 + 1) = 8/251 := by
  sorry

end problem_statement_l1573_157331


namespace functional_equation_solution_l1573_157320

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + f y) - f x = (x + f y)^4 - x^4) :
  (∀ x : ℝ, f x = 0) ∨ (∃ k : ℝ, ∀ x : ℝ, f x = x^4 + k) := by
  sorry

end functional_equation_solution_l1573_157320


namespace box_volume_increase_l1573_157319

theorem box_volume_increase (l w h : ℝ) : 
  l * w * h = 5000 →
  2 * (l * w + w * h + l * h) = 1850 →
  4 * (l + w + h) = 240 →
  (l + 3) * (w + 3) * (h + 3) = 8342 := by
sorry

end box_volume_increase_l1573_157319


namespace decimal_point_problem_l1573_157360

theorem decimal_point_problem :
  ∃ (x y : ℝ), y - x = 7.02 ∧ y = 10 * x ∧ x = 0.78 ∧ y = 7.8 := by
  sorry

end decimal_point_problem_l1573_157360


namespace operations_result_l1573_157324

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 2*a - 3*b

-- Theorem statement
theorem operations_result : T (S 8 3) 4 = 88 := by
  sorry

end operations_result_l1573_157324


namespace prove_weekly_pay_l1573_157389

def weekly_pay_problem (y_pay : ℝ) (x_percent : ℝ) : Prop :=
  let x_pay := x_percent * y_pay
  let total_pay := x_pay + y_pay
  y_pay = 263.64 ∧ x_percent = 1.2 → total_pay = 580.008

theorem prove_weekly_pay : weekly_pay_problem 263.64 1.2 := by
  sorry

end prove_weekly_pay_l1573_157389


namespace n_squared_divisible_by_144_l1573_157326

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) : 144 ∣ n^2 := by
  sorry

end n_squared_divisible_by_144_l1573_157326


namespace point_transformation_l1573_157303

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the transformations
def rotateY90 (p : Point3D) : Point3D :=
  { x := p.z, y := p.y, z := -p.x }

def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

def reflectXZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

-- Define the sequence of transformations
def transformSequence (p : Point3D) : Point3D :=
  p |> rotateY90
    |> reflectYZ
    |> reflectXZ
    |> rotateY90
    |> reflectXZ
    |> reflectXY

-- Theorem statement
theorem point_transformation :
  let initial := Point3D.mk 2 2 2
  transformSequence initial = Point3D.mk (-2) 2 (-2) := by
  sorry

end point_transformation_l1573_157303


namespace vector_angle_problem_l1573_157387

theorem vector_angle_problem (α β : Real) (a b : Fin 2 → Real) :
  a 0 = Real.cos α ∧ a 1 = Real.sin α ∧
  b 0 = Real.cos β ∧ b 1 = Real.sin β ∧
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 2 * Real.sqrt 5 / 5 ∧
  0 < α ∧ α < π / 2 ∧
  -π / 2 < β ∧ β < 0 ∧
  Real.sin β = -5 / 13 →
  Real.cos (α - β) = 3 / 5 ∧ Real.sin α = 33 / 65 := by
sorry

end vector_angle_problem_l1573_157387


namespace exam_attendance_l1573_157332

theorem exam_attendance (passed_percentage : ℚ) (failed_count : ℕ) : 
  passed_percentage = 35/100 →
  failed_count = 546 →
  (failed_count : ℚ) / (1 - passed_percentage) = 840 :=
by
  sorry

end exam_attendance_l1573_157332


namespace dartboard_central_angle_l1573_157381

/-- The measure of the central angle of one section in a circular dartboard -/
def central_angle_measure (num_sections : ℕ) (section_probability : ℚ) : ℚ :=
  360 * section_probability

/-- Theorem: The central angle measure for a circular dartboard with 8 equal sections
    and 1/8 probability of landing in each section is 45 degrees -/
theorem dartboard_central_angle :
  central_angle_measure 8 (1/8) = 45 := by
  sorry

end dartboard_central_angle_l1573_157381


namespace unique_root_continuous_monotonic_l1573_157323

theorem unique_root_continuous_monotonic {α : Type*} [LinearOrder α] [TopologicalSpace α] {f : α → ℝ} {a b : α} (h_cont : Continuous f) (h_mono : Monotone f) (h_sign : f a * f b < 0) : ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end unique_root_continuous_monotonic_l1573_157323


namespace interior_angle_regular_octagon_l1573_157311

theorem interior_angle_regular_octagon :
  let sum_exterior_angles : ℝ := 360
  let num_sides : ℕ := 8
  let exterior_angle : ℝ := sum_exterior_angles / num_sides
  let interior_angle : ℝ := 180 - exterior_angle
  interior_angle = 135 := by
sorry

end interior_angle_regular_octagon_l1573_157311


namespace perpendicular_planes_condition_l1573_157363

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset_line_plane : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- Theorem statement
theorem perpendicular_planes_condition 
  (h_distinct : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (∀ m, subset_line_plane m α → 
    (perpendicular_line_plane m β → perpendicular_plane_plane α β) ∧
    ¬(perpendicular_plane_plane α β → perpendicular_line_plane m β)) :=
by sorry

end perpendicular_planes_condition_l1573_157363


namespace specific_cistern_wet_area_l1573_157399

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the wet surface area of a specific cistern -/
theorem specific_cistern_wet_area :
  cisternWetArea 6 4 1.25 = 49 := by
  sorry

end specific_cistern_wet_area_l1573_157399


namespace probability_at_least_two_same_l1573_157382

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling 5 fair 8-sided dice -/
theorem probability_at_least_two_same (numSides : ℕ) (numDice : ℕ) :
  numSides = 8 → numDice = 5 →
  (1 - (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice : ℚ) = 6512 / 8192 := by
  sorry

end probability_at_least_two_same_l1573_157382


namespace two_zeros_implies_a_geq_two_l1573_157396

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2 * x - a else Real.log (1 - x)

theorem two_zeros_implies_a_geq_two (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(f a x = 0 ∧ f a y = 0 ∧ f a z = 0)) →
  a ≥ 2 :=
sorry

end two_zeros_implies_a_geq_two_l1573_157396


namespace jerry_max_throws_l1573_157383

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's current misbehavior record -/
structure MisbehaviorRecord where
  interrupts : ℕ
  insults : ℕ

/-- Calculates the maximum number of times Jerry can throw things before reaching the office threshold -/
def max_throws (ps : PointSystem) (record : MisbehaviorRecord) : ℕ :=
  let current_points := record.interrupts * ps.interrupt_points + record.insults * ps.insult_points
  let remaining_points := ps.office_threshold - current_points
  remaining_points / ps.throw_points

/-- Theorem stating that Jerry can throw things twice before being sent to the office -/
theorem jerry_max_throws :
  let ps : PointSystem := {
    interrupt_points := 5,
    insult_points := 10,
    throw_points := 25,
    office_threshold := 100
  }
  let record : MisbehaviorRecord := {
    interrupts := 2,
    insults := 4
  }
  max_throws ps record = 2 := by
  sorry

end jerry_max_throws_l1573_157383


namespace total_boxes_theorem_l1573_157329

/-- Calculates the total number of boxes sold over four days given specific sales conditions --/
def total_boxes_sold (thursday_boxes : ℕ) : ℕ :=
  let friday_boxes : ℕ := thursday_boxes + (thursday_boxes * 50) / 100
  let saturday_boxes : ℕ := friday_boxes + (friday_boxes * 80) / 100
  let sunday_boxes : ℕ := saturday_boxes - (saturday_boxes * 30) / 100
  thursday_boxes + friday_boxes + saturday_boxes + sunday_boxes

/-- Theorem stating that given the specific sales conditions, the total number of boxes sold is 425 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 425 := by
  sorry

end total_boxes_theorem_l1573_157329


namespace sum_of_reciprocal_relations_l1573_157318

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 1) 
  (h2 : 1/x - 1/y = 9) : 
  x + y = -1/20 := by sorry

end sum_of_reciprocal_relations_l1573_157318


namespace ellipse_equation_l1573_157376

/-- The standard equation of an ellipse passing through (-3, 2) with the same foci as x²/9 + y²/4 = 1 -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  ((-3)^2 / a^2 + 2^2 / b^2 = 1) ∧
  (a^2 - b^2 = 9 - 4) ∧
  (a^2 = 15 ∧ b^2 = 10) :=
by sorry

end ellipse_equation_l1573_157376


namespace complement_of_angle_A_l1573_157306

-- Define the angle A
def angle_A : ℝ := 76

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A : complement angle_A = 14 := by
  sorry

end complement_of_angle_A_l1573_157306


namespace fourth_guard_runs_150_meters_l1573_157378

/-- The length of the rectangle in meters -/
def length : ℝ := 200

/-- The width of the rectangle in meters -/
def width : ℝ := 300

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 2 * (length + width)

/-- The total distance run by three guards in meters -/
def three_guards_distance : ℝ := 850

/-- The distance run by the fourth guard in meters -/
def fourth_guard_distance : ℝ := perimeter - three_guards_distance

theorem fourth_guard_runs_150_meters :
  fourth_guard_distance = 150 := by sorry

end fourth_guard_runs_150_meters_l1573_157378


namespace quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l1573_157321

-- Define a quadrangle
structure Quadrangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- State the theorem
theorem quadrangle_area_inequality (q : Quadrangle) :
  q.area ≤ (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- Define a convex orthodiagonal cyclic quadrilateral
structure ConvexOrthoDiagonalCyclicQuad extends Quadrangle where
  is_convex : Bool
  is_orthodiagonal : Bool
  is_cyclic : Bool

-- State the equality condition
theorem quadrangle_area_equality (q : ConvexOrthoDiagonalCyclicQuad) :
  q.is_convex = true → q.is_orthodiagonal = true → q.is_cyclic = true →
  q.area = (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- State the converse of the equality condition
theorem quadrangle_area_equality_converse (q : Quadrangle) :
  q.area = (1/2) * (q.a * q.c + q.b * q.d) →
  ∃ (cq : ConvexOrthoDiagonalCyclicQuad),
    cq.a = q.a ∧ cq.b = q.b ∧ cq.c = q.c ∧ cq.d = q.d ∧
    cq.area = q.area ∧
    cq.is_convex = true ∧ cq.is_orthodiagonal = true ∧ cq.is_cyclic = true := by sorry

end quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l1573_157321


namespace arthur_baked_115_muffins_l1573_157356

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The number of muffins James baked -/
def james_muffins : ℕ := 1380

/-- James baked 12 times as many muffins as Arthur -/
axiom james_baked_12_times : james_muffins = 12 * arthur_muffins

theorem arthur_baked_115_muffins : arthur_muffins = 115 := by
  sorry

end arthur_baked_115_muffins_l1573_157356


namespace total_fish_count_l1573_157312

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
sorry

end total_fish_count_l1573_157312


namespace max_k_logarithmic_inequality_l1573_157338

theorem max_k_logarithmic_inequality (x₀ x₁ x₂ x₃ : ℝ) (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  ∃ k : ℝ, k = 9 ∧ 
  ∀ k' : ℝ, k' > k → 
  ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
  (Real.log (x₀' / x₁') / Real.log (x₀ / x₁) + 
   Real.log (x₁' / x₂') / Real.log (x₁ / x₂) + 
   Real.log (x₂' / x₃') / Real.log (x₂ / x₃) ≤ 
   k' * Real.log (x₀' / x₃') / Real.log (x₀ / x₃)) :=
by sorry

end max_k_logarithmic_inequality_l1573_157338


namespace solve_equation_l1573_157372

theorem solve_equation (p q x : ℚ) : 
  (3 / 4 : ℚ) = p / 60 ∧ 
  (3 / 4 : ℚ) = (p + q) / 100 ∧ 
  (3 / 4 : ℚ) = (x - q) / 140 → 
  x = 135 := by sorry

end solve_equation_l1573_157372


namespace exists_valid_coloring_l1573_157341

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define the property of a color appearing on infinitely many lines
def InfiniteLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f (x, y) = c

-- Define the parallelogram property
def ParallelogramProperty (f : ColoringFunction) : Prop :=
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (InfiniteLines f Color.White) ∧
  (InfiniteLines f Color.Red) ∧
  (InfiniteLines f Color.Black) ∧
  ParallelogramProperty f :=
sorry

end exists_valid_coloring_l1573_157341


namespace line_vertical_translation_l1573_157316

/-- The equation of a line after vertical translation -/
theorem line_vertical_translation (x y : ℝ) :
  (y = x) → (y = x + 2) ↔ (∀ point : ℝ × ℝ, point.2 = point.1 + 2 ↔ point.2 = point.1 + 2) :=
by sorry

end line_vertical_translation_l1573_157316


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1573_157350

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  r₁ + r₂ = 8 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1573_157350


namespace sum_of_m_and_n_l1573_157342

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6*m + 10*n + 34 = 0) : m + n = -2 := by
  sorry

end sum_of_m_and_n_l1573_157342


namespace convex_polygon_mean_inequality_l1573_157335

/-- For a convex n-gon, the arithmetic mean of side lengths is less than the arithmetic mean of diagonal lengths -/
theorem convex_polygon_mean_inequality {n : ℕ} (hn : n ≥ 3) 
  (P : ℝ) (D : ℝ) (hP : P > 0) (hD : D > 0) :
  P / n < (2 * D) / (n * (n - 3)) := by
  sorry

end convex_polygon_mean_inequality_l1573_157335


namespace maciek_purchase_cost_l1573_157379

-- Define the pricing structure
def pretzel_price (quantity : ℕ) : ℚ :=
  if quantity > 4 then 3.5 else 4

def chip_price (quantity : ℕ) : ℚ :=
  if quantity > 3 then 6.5 else 7

def soda_price (quantity : ℕ) : ℚ :=
  if quantity > 5 then 1.5 else 2

-- Define Maciek's purchase quantities
def pretzel_quantity : ℕ := 5
def chip_quantity : ℕ := 4
def soda_quantity : ℕ := 6

-- Calculate the total cost
def total_cost : ℚ :=
  pretzel_price pretzel_quantity * pretzel_quantity +
  chip_price chip_quantity * chip_quantity +
  soda_price soda_quantity * soda_quantity

-- Theorem statement
theorem maciek_purchase_cost :
  total_cost = 52.5 := by sorry

end maciek_purchase_cost_l1573_157379


namespace abs_inequality_iff_gt_l1573_157347

theorem abs_inequality_iff_gt (a b : ℝ) (h : a * b > 0) :
  a * |a| > b * |b| ↔ a > b :=
sorry

end abs_inequality_iff_gt_l1573_157347


namespace triangle_problem_l1573_157349

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (a + c = 6) →
  (b = 2) →
  (Real.cos B = 7/9) →
  (a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A))) →
  (b = Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))) →
  (c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C))) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A + B + C = Real.pi) →
  (a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27) := by
  sorry


end triangle_problem_l1573_157349


namespace x_value_proof_l1573_157330

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 6)
  (h3 : z^3 / x = 9) :
  x = (559872 : ℝ) ^ (1 / 38) :=
by sorry

end x_value_proof_l1573_157330


namespace M_subset_N_l1573_157375

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {x | x ≤ 1}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l1573_157375


namespace sum_of_solutions_l1573_157310

theorem sum_of_solutions (x : ℝ) : 
  (x^2 + 2023*x = 2025) → 
  (∃ y : ℝ, y^2 + 2023*y = 2025 ∧ x + y = -2023) :=
by sorry

end sum_of_solutions_l1573_157310


namespace sum_of_products_l1573_157301

theorem sum_of_products (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + x*z + x^2 = 91) :
  x*y + y*z + x*z = 40 := by
sorry

end sum_of_products_l1573_157301


namespace quadratic_inequality_solution_l1573_157386

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 3*x - 10 > 0) ↔ (x < -2 ∨ x > 5) := by
  sorry

end quadratic_inequality_solution_l1573_157386


namespace trick_decks_spending_l1573_157307

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem: Frank and his friend spent 35 dollars on trick decks -/
theorem trick_decks_spending : total_spent 7 3 2 = 35 := by
  sorry

end trick_decks_spending_l1573_157307


namespace corresponding_angles_equal_l1573_157344

theorem corresponding_angles_equal (α β γ : ℝ) : 
  α + β + γ = 180 → 
  (180 - α) + β + γ = 180 → 
  α = 180 - α ∧ β = β ∧ γ = γ := by
sorry

end corresponding_angles_equal_l1573_157344


namespace sum_interior_angles_30_vertices_l1573_157339

/-- The sum of interior angles of faces in a convex polyhedron with given number of vertices -/
def sum_interior_angles (vertices : ℕ) : ℝ :=
  (vertices - 2) * 180

/-- Theorem: The sum of interior angles of faces in a convex polyhedron with 30 vertices is 5040° -/
theorem sum_interior_angles_30_vertices :
  sum_interior_angles 30 = 5040 := by
  sorry

end sum_interior_angles_30_vertices_l1573_157339


namespace total_books_proof_l1573_157328

/-- The number of books taken by the librarian -/
def books_taken_by_librarian : ℕ := 7

/-- The number of books Jerry can fit on one shelf -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed + books_taken_by_librarian

theorem total_books_proof : total_books = 34 := by
  sorry

end total_books_proof_l1573_157328


namespace ramp_cost_calculation_l1573_157337

def ramp_installation_cost (permits_cost : ℝ) (contractor_labor_rate : ℝ) 
  (contractor_materials_rate : ℝ) (contractor_days : ℕ) (contractor_hours_per_day : ℝ) 
  (contractor_lunch_break : ℝ) (inspector_rate_discount : ℝ) (inspector_hours_per_day : ℝ) : ℝ :=
  let contractor_work_hours := (contractor_hours_per_day - contractor_lunch_break) * contractor_days
  let contractor_labor_cost := contractor_work_hours * contractor_labor_rate
  let materials_cost := contractor_work_hours * contractor_materials_rate
  let inspector_rate := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost := inspector_rate * inspector_hours_per_day * contractor_days
  permits_cost + contractor_labor_cost + materials_cost + inspector_cost

theorem ramp_cost_calculation :
  ramp_installation_cost 250 150 50 3 5 0.5 0.8 2 = 3130 := by
  sorry

end ramp_cost_calculation_l1573_157337


namespace rebus_solution_l1573_157392

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A + B * 10 + B = C * 10 + A ∧
    C = 1 ∧ B = 9 ∧ A = 6 := by
  sorry

end rebus_solution_l1573_157392


namespace intersection_when_a_is_quarter_b_necessary_condition_for_a_l1573_157309

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 1}

-- Theorem for part 1
theorem intersection_when_a_is_quarter :
  A ∩ B (1/4) = {x | 1 < x ∧ x < 7/4} := by sorry

-- Theorem for part 2
theorem b_necessary_condition_for_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ↔ 1/3 ≤ a ∧ a ≤ 2 := by sorry

end intersection_when_a_is_quarter_b_necessary_condition_for_a_l1573_157309


namespace crow_speed_l1573_157358

/-- Calculates the speed of a crow flying between its nest and a ditch -/
theorem crow_speed (distance : ℝ) (trips : ℕ) (time : ℝ) : 
  distance = 200 → 
  trips = 15 → 
  time = 1.5 → 
  (2 * distance * trips) / (time * 1000) = 4 :=
by sorry

end crow_speed_l1573_157358


namespace smallest_k_with_remainder_one_l1573_157371

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_k_with_remainder_one_l1573_157371


namespace team_average_score_l1573_157359

theorem team_average_score (player1 player2 player3 player4 : ℝ) 
  (h1 : player1 = 20)
  (h2 : player2 = player1 / 2)
  (h3 : player3 = 6 * player2)
  (h4 : player4 = 3 * player3) :
  (player1 + player2 + player3 + player4) / 4 = 67.5 := by
  sorry

end team_average_score_l1573_157359


namespace rope_division_l1573_157340

theorem rope_division (initial_length : ℝ) (initial_cuts : ℕ) (final_cuts : ℕ) : 
  initial_length = 200 →
  initial_cuts = 4 →
  final_cuts = 2 →
  (initial_length / initial_cuts) / final_cuts = 25 := by
  sorry

end rope_division_l1573_157340


namespace trapezoid_area_l1573_157346

/-- The area of a trapezoid bounded by y = 2x, y = 10, y = 5, and the y-axis -/
theorem trapezoid_area : ∃ (A : ℝ), A = 18.75 ∧ 
  A = ((5 - 0) + (10 - 5)) / 2 * 5 ∧
  (∀ x y : ℝ, (y = 2*x ∨ y = 10 ∨ y = 5 ∨ x = 0) → 
    0 ≤ x ∧ x ≤ 5 ∧ 5 ≤ y ∧ y ≤ 10) :=
by sorry

end trapezoid_area_l1573_157346


namespace square_difference_from_sum_and_difference_l1573_157377

theorem square_difference_from_sum_and_difference (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end square_difference_from_sum_and_difference_l1573_157377


namespace parallel_vectors_x_value_l1573_157373

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
sorry

end parallel_vectors_x_value_l1573_157373


namespace number_of_boys_l1573_157302

theorem number_of_boys (total : ℕ) (x : ℕ) : 
  total = 150 → 
  x + (x * total) / 100 = total → 
  x = 60 := by
sorry

end number_of_boys_l1573_157302


namespace discount_savings_l1573_157388

/-- Given a store with an 8% discount and a customer who pays $184 for an item, 
    prove that the amount saved is $16. -/
theorem discount_savings (discount_rate : ℝ) (paid_amount : ℝ) (saved_amount : ℝ) : 
  discount_rate = 0.08 →
  paid_amount = 184 →
  saved_amount = 16 →
  paid_amount / (1 - discount_rate) * discount_rate = saved_amount := by
sorry

end discount_savings_l1573_157388


namespace magnitude_of_3_plus_i_squared_l1573_157345

theorem magnitude_of_3_plus_i_squared : 
  Complex.abs ((3 : ℂ) + Complex.I) ^ 2 = 10 := by
  sorry

end magnitude_of_3_plus_i_squared_l1573_157345


namespace quadratic_inequality_l1573_157325

theorem quadratic_inequality (d : ℝ) : 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 1) ↔ d = -5 := by
sorry

end quadratic_inequality_l1573_157325


namespace complement_intersection_M_N_l1573_157368

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 5} := by sorry

end complement_intersection_M_N_l1573_157368


namespace arithmetic_mean_problem_l1573_157366

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 21 + 7 + 12 + y) / 6 = 15 → y = 27 := by
sorry

end arithmetic_mean_problem_l1573_157366


namespace find_a_value_l1573_157357

theorem find_a_value (A B : Set ℝ) (a : ℝ) :
  A = {2^a, 3} →
  B = {2, 3} →
  A ∪ B = {1, 2, 3} →
  a = 0 := by
sorry

end find_a_value_l1573_157357


namespace lcm_of_36_and_132_l1573_157308

theorem lcm_of_36_and_132 (hcf : ℕ) (lcm : ℕ) :
  hcf = 12 →
  lcm = 36 * 132 / hcf →
  lcm = 396 := by
sorry

end lcm_of_36_and_132_l1573_157308


namespace perfect_square_solutions_l1573_157343

theorem perfect_square_solutions : 
  {n : ℤ | ∃ k : ℤ, n^2 + 8*n + 44 = k^2} = {2, -10} := by sorry

end perfect_square_solutions_l1573_157343


namespace total_cost_is_94_l1573_157305

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (cost : GoodsCost) : Prop :=
  cost.A + 2 * cost.B + 3 * cost.C = 136 ∧
  3 * cost.A + 2 * cost.B + cost.C = 240

/-- The theorem to be proved -/
theorem total_cost_is_94 (cost : GoodsCost) (h : satisfies_conditions cost) : 
  cost.A + cost.B + cost.C = 94 := by
  sorry

end total_cost_is_94_l1573_157305


namespace max_modest_number_l1573_157334

def is_modest_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  5 * a = b + c + d

def G (a b c d : ℕ) : ℤ :=
  10 * a + b - 10 * c - d

theorem max_modest_number :
  ∀ (a b c d : ℕ),
    is_modest_number a b c d →
    d % 2 = 0 →
    (G a b c d) % 11 = 0 →
    (a + b + c) % 3 = 0 →
    a * 1000 + b * 100 + c * 10 + d ≤ 3816 :=
by sorry

end max_modest_number_l1573_157334


namespace negation_of_positive_square_plus_two_is_false_l1573_157336

theorem negation_of_positive_square_plus_two_is_false : 
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) := by
sorry

end negation_of_positive_square_plus_two_is_false_l1573_157336


namespace tan_alpha_negative_three_l1573_157314

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 ∧
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := by
  sorry

end tan_alpha_negative_three_l1573_157314


namespace consecutive_integers_sum_l1573_157364

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 506) : 
  x + (x + 1) = 45 := by
  sorry

end consecutive_integers_sum_l1573_157364


namespace circle_circumference_l1573_157390

theorem circle_circumference (r : ℝ) (d : ℝ) (C : ℝ) :
  (d = 2 * r) → (C = π * d ∨ C = 2 * π * r) :=
by sorry

end circle_circumference_l1573_157390


namespace girls_in_class_l1573_157370

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (ratio_girls * total) / (ratio_girls + ratio_boys) = 15 := by
sorry

end girls_in_class_l1573_157370


namespace P_lower_bound_and_equality_l1573_157355

/-- The number of 4k-digit numbers composed of digits 2 and 0 (not starting with 0) that are divisible by 2020 -/
def P (k : ℕ+) : ℕ := sorry

/-- The theorem stating the inequality and the condition for equality -/
theorem P_lower_bound_and_equality (k : ℕ+) :
  P k ≥ Nat.choose (2 * k - 1) k ^ 2 ∧
  (P k = Nat.choose (2 * k - 1) k ^ 2 ↔ k ≤ 9) :=
sorry

end P_lower_bound_and_equality_l1573_157355


namespace sum_of_parts_l1573_157374

theorem sum_of_parts (x y : ℤ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end sum_of_parts_l1573_157374


namespace initial_diaries_count_l1573_157351

theorem initial_diaries_count (initial : ℕ) : 
  (2 * initial - (2 * initial) / 4 = 18) → initial = 12 := by
  sorry

end initial_diaries_count_l1573_157351


namespace bicycle_cost_price_l1573_157348

/-- The cost price of the bicycle for A -/
def cost_price_A : ℝ := sorry

/-- The selling price from A to B -/
def selling_price_B : ℝ := 1.20 * cost_price_A

/-- The selling price from B to C before tax -/
def selling_price_C : ℝ := 1.25 * selling_price_B

/-- The total cost for C including tax -/
def total_cost_C : ℝ := 1.15 * selling_price_C

/-- The selling price from C to D before discount -/
def selling_price_D1 : ℝ := 1.30 * total_cost_C

/-- The final selling price from C to D after discount -/
def selling_price_D2 : ℝ := 0.90 * selling_price_D1

/-- The final price D pays for the bicycle -/
def final_price_D : ℝ := 350

theorem bicycle_cost_price :
  cost_price_A = final_price_D / 2.01825 := by sorry

end bicycle_cost_price_l1573_157348


namespace sin_300_degrees_l1573_157313

theorem sin_300_degrees : 
  Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l1573_157313


namespace terms_before_negative_twenty_l1573_157367

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_negative_twenty :
  let a₁ := 100
  let d := -4
  let n := 31
  arithmetic_sequence a₁ d n = -20 ∧ n - 1 = 30 := by
  sorry

end terms_before_negative_twenty_l1573_157367


namespace slope_condition_implies_y_value_l1573_157353

/-- Given two points P and Q in a coordinate plane, where P has coordinates (-3, 5) and Q has coordinates (5, y), prove that if the slope of the line through P and Q is -4/3, then y = -17/3. -/
theorem slope_condition_implies_y_value :
  let P : ℝ × ℝ := (-3, 5)
  let Q : ℝ × ℝ := (5, y)
  let slope := (Q.2 - P.2) / (Q.1 - P.1)
  slope = -4/3 → y = -17/3 :=
by
  sorry

end slope_condition_implies_y_value_l1573_157353
