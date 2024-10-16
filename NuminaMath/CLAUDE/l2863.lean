import Mathlib

namespace NUMINAMATH_CALUDE_tims_drive_distance_l2863_286313

/-- Represents the scenario of Tim's drive to work -/
def TimsDrive (totalDistance : ℝ) : Prop :=
  let normalTime : ℝ := 120
  let newTime : ℝ := 165
  let speedReduction : ℝ := 30 / 60 -- 30 mph converted to miles per minute
  let normalSpeed : ℝ := totalDistance / normalTime
  let newSpeed : ℝ := normalSpeed - speedReduction
  let halfDistance : ℝ := totalDistance / 2
  normalTime / 2 + halfDistance / newSpeed = newTime

/-- Theorem stating that the total distance of Tim's drive is 140 miles -/
theorem tims_drive_distance : ∃ (d : ℝ), TimsDrive d ∧ d = 140 :=
sorry

end NUMINAMATH_CALUDE_tims_drive_distance_l2863_286313


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2863_286388

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- side lengths

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.B = 60 ∧ t.b^2 = t.a * t.c

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) (h : is_valid_triangle t) : 
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2863_286388


namespace NUMINAMATH_CALUDE_course_selection_methods_l2863_286389

/-- The number of courses in Group A -/
def group_A_courses : ℕ := 3

/-- The number of courses in Group B -/
def group_B_courses : ℕ := 4

/-- The total number of courses that must be selected -/
def total_selected : ℕ := 3

/-- The function to calculate the number of ways to select courses -/
def select_courses (group_A : ℕ) (group_B : ℕ) (total : ℕ) : ℕ :=
  Nat.choose group_A 2 * Nat.choose group_B 1 +
  Nat.choose group_A 1 * Nat.choose group_B 2

/-- Theorem stating that the number of different selection methods is 30 -/
theorem course_selection_methods :
  select_courses group_A_courses group_B_courses total_selected = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_methods_l2863_286389


namespace NUMINAMATH_CALUDE_total_animals_l2863_286374

theorem total_animals (giraffes pigs dogs : ℕ) 
  (h1 : giraffes = 6) 
  (h2 : pigs = 8) 
  (h3 : dogs = 4) : 
  giraffes + pigs + dogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l2863_286374


namespace NUMINAMATH_CALUDE_milk_container_percentage_difference_l2863_286342

/-- Given a scenario where milk is transferred between containers, this theorem proves
    the percentage difference between the quantity in one container and the original capacity. -/
theorem milk_container_percentage_difference
  (total_milk : ℝ)
  (transfer_amount : ℝ)
  (h_total : total_milk = 1216)
  (h_transfer : transfer_amount = 152)
  (h_equal_after_transfer : ∃ (b c : ℝ), b + c = total_milk ∧ b + transfer_amount = c - transfer_amount) :
  ∃ (b : ℝ), (total_milk - b) / total_milk * 100 = 56.25 := by
  sorry

#eval (1216 - 532) / 1216 * 100  -- Should output approximately 56.25

end NUMINAMATH_CALUDE_milk_container_percentage_difference_l2863_286342


namespace NUMINAMATH_CALUDE_cos_240_degrees_l2863_286349

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l2863_286349


namespace NUMINAMATH_CALUDE_inequality_proof_l2863_286327

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2863_286327


namespace NUMINAMATH_CALUDE_bus_station_arrangement_count_l2863_286375

/-- The number of seats in the bus station -/
def num_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- The number of ways to arrange passengers with the required consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- Theorem stating that the number of ways to arrange passengers with the required consecutive empty seats is correct -/
theorem bus_station_arrangement_count :
  (num_seats : ℕ) = 10 →
  (num_passengers : ℕ) = 4 →
  (consecutive_empty_seats : ℕ) = 5 →
  (arrangement_count : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_bus_station_arrangement_count_l2863_286375


namespace NUMINAMATH_CALUDE_brownies_before_division_l2863_286314

def initial_brownies : ℕ := 24  -- 2 dozen

def father_ate (n : ℕ) : ℕ := n / 3

def mooney_ate (n : ℕ) : ℕ := n / 4  -- 25% = 1/4

def benny_ate (n : ℕ) : ℕ := n * 2 / 5

def snoopy_ate : ℕ := 3

def mother_baked_wednesday : ℕ := 18  -- 1.5 dozen

def mother_baked_thursday : ℕ := 36  -- 3 dozen

def final_brownies : ℕ :=
  let after_father := initial_brownies - father_ate initial_brownies
  let after_mooney := after_father - mooney_ate after_father
  let after_benny := after_mooney - benny_ate after_mooney
  let after_snoopy := after_benny - snoopy_ate
  after_snoopy + mother_baked_wednesday + mother_baked_thursday

theorem brownies_before_division :
  final_brownies = 59 := by sorry

end NUMINAMATH_CALUDE_brownies_before_division_l2863_286314


namespace NUMINAMATH_CALUDE_sum_of_ages_l2863_286308

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett olaf shannen jack : ℕ) : 
  beckett = 12 ∧ 
  olaf = beckett + 3 ∧ 
  shannen = olaf - 2 ∧ 
  jack = 2 * shannen + 5 → 
  beckett + olaf + shannen + jack = 71 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2863_286308


namespace NUMINAMATH_CALUDE_product_of_reals_l2863_286360

theorem product_of_reals (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l2863_286360


namespace NUMINAMATH_CALUDE_circle_placement_theorem_l2863_286317

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a circle with given diameter -/
structure Circle where
  diameter : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point for a circle with diameter 1 -/
theorem circle_placement_theorem (rect : Rectangle) (squares : Finset Square) (circ : Circle) :
  rect.width = 20 ∧ rect.height = 25 ∧
  squares.card = 120 ∧ (∀ s ∈ squares, s.sideLength = 1) ∧
  circ.diameter = 1 →
  ∃ (center : ℝ × ℝ),
    (center.1 ≥ 0 ∧ center.1 ≤ rect.width ∧ center.2 ≥ 0 ∧ center.2 ≤ rect.height) ∧
    (∀ s ∈ squares, ∀ (point : ℝ × ℝ),
      (point.1 - center.1)^2 + (point.2 - center.2)^2 ≤ (circ.diameter / 2)^2 →
      ¬(point.1 ≥ s.sideLength ∧ point.1 ≤ s.sideLength + 1 ∧
        point.2 ≥ s.sideLength ∧ point.2 ≤ s.sideLength + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_placement_theorem_l2863_286317


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2863_286370

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {2, 3}
def B : Set Int := {0, 1}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2863_286370


namespace NUMINAMATH_CALUDE_square_carpet_side_length_l2863_286334

theorem square_carpet_side_length 
  (floor_length : ℝ) 
  (floor_width : ℝ) 
  (uncovered_area : ℝ) 
  (h1 : floor_length = 10)
  (h2 : floor_width = 8)
  (h3 : uncovered_area = 64)
  : ∃ (side_length : ℝ), 
    side_length^2 = floor_length * floor_width - uncovered_area ∧ 
    side_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_carpet_side_length_l2863_286334


namespace NUMINAMATH_CALUDE_parabola_from_hyperbola_l2863_286309

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  (positive_p : 0 < p)

/-- The center of a hyperbola -/
def Hyperbola.center (h : Hyperbola) : ℝ × ℝ := (0, 0)

/-- The right focus of a hyperbola -/
def Hyperbola.right_focus (h : Hyperbola) : ℝ × ℝ := (h.a, 0)

/-- The equation of a parabola with vertex at the origin -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y^2 = 4 * p.p * x

theorem parabola_from_hyperbola (h : Hyperbola) 
    (h_eq : ∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ (x / h.a)^2 - (y / h.b)^2 = 1) :
    ∃ p : Parabola, 
      p.equation = fun x y => y^2 = 12 * x ∧
      Parabola.equation p x y ↔ y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_from_hyperbola_l2863_286309


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2863_286362

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubseq (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sum (a : ℕ → ℤ) 
  (h_arith : arithmeticSeq a) (h_geom : geometricSubseq a) : 
  a 2 + a 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2863_286362


namespace NUMINAMATH_CALUDE_exists_g_for_f_l2863_286390

-- Define the function f: ℝ² → ℝ
variable (f : ℝ × ℝ → ℝ)

-- State the condition for f
axiom f_condition : ∀ (x y z : ℝ), f (x, y) + f (y, z) + f (z, x) = 0

-- Theorem statement
theorem exists_g_for_f : 
  ∃ (g : ℝ → ℝ), ∀ (x y : ℝ), f (x, y) = g x - g y := by sorry

end NUMINAMATH_CALUDE_exists_g_for_f_l2863_286390


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2863_286333

theorem imaginary_part_of_z (z : ℂ) (h : z + Complex.abs z = 1 + 2*I) : 
  Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2863_286333


namespace NUMINAMATH_CALUDE_horner_v2_value_l2863_286346

/-- Horner's method for polynomial evaluation -/
def horner_step (x : ℝ) (a : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = x^4 + 2x^3 - 3x^2 + x + 5 -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - 3*x^2 + x + 5

theorem horner_v2_value :
  let x : ℝ := 2
  let v₁ : ℝ := horner_step x 2 1  -- Corresponds to x + 2
  let v₂ : ℝ := horner_step x (-3) v₁  -- Corresponds to v₁ * x - 3
  v₂ = 5 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l2863_286346


namespace NUMINAMATH_CALUDE_pet_shelter_problem_l2863_286357

theorem pet_shelter_problem (total dogs_watermelon dogs_salmon dogs_chicken 
  dogs_watermelon_salmon dogs_salmon_chicken dogs_watermelon_chicken dogs_all_three : ℕ) 
  (h_total : total = 150)
  (h_watermelon : dogs_watermelon = 30)
  (h_salmon : dogs_salmon = 70)
  (h_chicken : dogs_chicken = 15)
  (h_watermelon_salmon : dogs_watermelon_salmon = 10)
  (h_salmon_chicken : dogs_salmon_chicken = 7)
  (h_watermelon_chicken : dogs_watermelon_chicken = 5)
  (h_all_three : dogs_all_three = 3) :
  total - (dogs_watermelon + dogs_salmon + dogs_chicken 
    - dogs_watermelon_salmon - dogs_salmon_chicken - dogs_watermelon_chicken 
    + dogs_all_three) = 54 := by
  sorry


end NUMINAMATH_CALUDE_pet_shelter_problem_l2863_286357


namespace NUMINAMATH_CALUDE_a_4_equals_8_l2863_286382

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem a_4_equals_8 (a : ℕ → ℝ) 
    (h1 : a 1 = 1)
    (h2 : ∀ (n : ℕ), a (n + 1) = 2 * a n) : 
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l2863_286382


namespace NUMINAMATH_CALUDE_alice_savings_this_month_l2863_286367

/-- Alice's sales and earnings calculation --/
def alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let commission := sales * commission_rate
  let total_earnings := basic_salary + commission
  total_earnings * savings_rate

/-- Theorem: Alice's savings this month will be $29 --/
theorem alice_savings_this_month :
  alice_savings 2500 240 0.02 0.10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_this_month_l2863_286367


namespace NUMINAMATH_CALUDE_probability_all_sides_of_decagon_l2863_286348

/-- A regular decagon --/
structure RegularDecagon where

/-- A triangle formed from three vertices of a regular decagon --/
structure DecagonTriangle where
  decagon : RegularDecagon
  vertex1 : Nat
  vertex2 : Nat
  vertex3 : Nat

/-- Predicate to check if three vertices are sequentially adjacent in a decagon --/
def are_sequential_adjacent (v1 v2 v3 : Nat) : Prop :=
  (v2 = (v1 + 1) % 10) ∧ (v3 = (v2 + 1) % 10)

/-- Predicate to check if a triangle's sides are all sides of the decagon --/
def all_sides_of_decagon (t : DecagonTriangle) : Prop :=
  are_sequential_adjacent t.vertex1 t.vertex2 t.vertex3

/-- The total number of possible triangles in a decagon --/
def total_triangles : Nat := 120

/-- The number of triangles with all sides being sides of the decagon --/
def favorable_triangles : Nat := 10

/-- The main theorem --/
theorem probability_all_sides_of_decagon :
  (favorable_triangles : ℚ) / total_triangles = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_sides_of_decagon_l2863_286348


namespace NUMINAMATH_CALUDE_intersection_circle_regions_l2863_286321

/-- The maximum number of regions in the intersection of n circles -/
def max_regions (n : ℕ) : ℕ :=
  2 * n - 2

/-- Theorem stating the maximum number of regions in the intersection of n circles -/
theorem intersection_circle_regions (n : ℕ) (h : n ≥ 2) :
  max_regions n = 2 * n - 2 := by
  sorry

#check intersection_circle_regions

end NUMINAMATH_CALUDE_intersection_circle_regions_l2863_286321


namespace NUMINAMATH_CALUDE_lower_rent_amount_l2863_286364

theorem lower_rent_amount (total_rent : ℕ) (higher_rent : ℕ) (num_rooms_changed : ℕ) 
  (h1 : total_rent = 1000)
  (h2 : higher_rent = 60)
  (h3 : num_rooms_changed = 10)
  (h4 : ∃ (lower_rent : ℕ), 
    ∃ (num_higher_rooms num_lower_rooms : ℕ),
      total_rent = higher_rent * num_higher_rooms + lower_rent * num_lower_rooms ∧
      total_rent * 4/5 = higher_rent * (num_higher_rooms - num_rooms_changed) + 
                         lower_rent * (num_lower_rooms + num_rooms_changed)) :
  ∃ (lower_rent : ℕ), lower_rent = 40 := by
sorry

end NUMINAMATH_CALUDE_lower_rent_amount_l2863_286364


namespace NUMINAMATH_CALUDE_stew_consumption_l2863_286325

theorem stew_consumption (total_stew : ℝ) : 
  let camp_fraction : ℝ := 1/3
  let range_fraction : ℝ := 1 - camp_fraction
  let lunch_consumption : ℝ := (1/4) * total_stew
  let evening_portion_multiplier : ℝ := 3/2
  let evening_consumption : ℝ := evening_portion_multiplier * (range_fraction / camp_fraction) * lunch_consumption
  lunch_consumption + evening_consumption = total_stew :=
by sorry

end NUMINAMATH_CALUDE_stew_consumption_l2863_286325


namespace NUMINAMATH_CALUDE_range_of_a_l2863_286377

/-- The function f(x) = |x+a| + |x-2| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

/-- The solution set A for f(x) ≤ |x-4| -/
def A (a : ℝ) : Set ℝ := {x | f a x ≤ |x - 4|}

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2863_286377


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2863_286301

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/3
def prob_trap : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l2863_286301


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2010_l2863_286337

theorem units_digit_of_7_to_2010 : (7^2010) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2010_l2863_286337


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2863_286355

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2863_286355


namespace NUMINAMATH_CALUDE_parallelepiped_diagonals_edges_squares_sum_equal_l2863_286329

/-- A parallelepiped with side lengths a, b, and c. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The sum of squares of the lengths of the four diagonals of a parallelepiped. -/
def sum_squares_diagonals (p : Parallelepiped) : ℝ :=
  4 * (p.a^2 + p.b^2 + p.c^2)

/-- The sum of squares of the lengths of the twelve edges of a parallelepiped. -/
def sum_squares_edges (p : Parallelepiped) : ℝ :=
  4 * p.a^2 + 4 * p.b^2 + 4 * p.c^2

/-- 
Theorem: The sum of the squares of the lengths of the four diagonals 
of a parallelepiped is equal to the sum of the squares of the lengths of its twelve edges.
-/
theorem parallelepiped_diagonals_edges_squares_sum_equal (p : Parallelepiped) :
  sum_squares_diagonals p = sum_squares_edges p := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonals_edges_squares_sum_equal_l2863_286329


namespace NUMINAMATH_CALUDE_derivative_of_exp_2x_l2863_286383

theorem derivative_of_exp_2x (x : ℝ) :
  deriv (fun x => Real.exp (2 * x)) x = 2 * Real.exp (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_2x_l2863_286383


namespace NUMINAMATH_CALUDE_trillion_to_scientific_notation_l2863_286302

/-- Represents the value of one trillion -/
def trillion : ℕ := 1000000000000

/-- Proves that 6.13 trillion is equal to 6.13 × 10^12 -/
theorem trillion_to_scientific_notation : 
  (6.13 : ℝ) * (trillion : ℝ) = 6.13 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_trillion_to_scientific_notation_l2863_286302


namespace NUMINAMATH_CALUDE_empty_cell_exists_l2863_286304

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- A function that checks if two cells are adjacent --/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Represents the movement of bugs --/
def moves (before after : Grid) : Prop :=
  ∀ (i j : Fin 5), 
    before i j → ∃ (i' j' : Fin 5), adjacent (i, j) (i', j') ∧ after i' j'

/-- The main theorem --/
theorem empty_cell_exists (before after : Grid) 
  (h1 : ∀ (i j : Fin 5), before i j)
  (h2 : moves before after) : 
  ∃ (i j : Fin 5), ¬after i j :=
sorry

end NUMINAMATH_CALUDE_empty_cell_exists_l2863_286304


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2863_286341

theorem age_ratio_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : a - 4 = b + 4) :
  3 * (b - 4) = a + 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2863_286341


namespace NUMINAMATH_CALUDE_moses_percentage_l2863_286376

theorem moses_percentage (total : ℝ) (moses_amount : ℝ) (esther_amount : ℝ) : 
  total = 50 ∧
  moses_amount = esther_amount + 5 ∧
  moses_amount + 2 * esther_amount = total →
  moses_amount / total = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_moses_percentage_l2863_286376


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l2863_286358

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bike : Nat
  truck : Nat
  bus : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bikes : Nat
  trucks : Nat
  buses : Nat

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (wheels : VehicleWheels) (count : VehicleCount) : Nat :=
  wheels.car * count.cars +
  wheels.bike * count.bikes +
  wheels.truck * count.trucks +
  wheels.bus * count.buses

/-- Theorem: The total number of wheels in the parking lot is 156 -/
theorem parking_lot_wheels :
  let wheels : VehicleWheels := ⟨4, 2, 8, 6⟩
  let count : VehicleCount := ⟨14, 10, 7, 4⟩
  totalWheels wheels count = 156 := by
  sorry

#check parking_lot_wheels

end NUMINAMATH_CALUDE_parking_lot_wheels_l2863_286358


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2863_286300

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- A line passing through a given point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_and_line_theorem (C : Ellipse) (l : Line) : 
  C.center = (0, 0) ∧ 
  C.foci_on_x_axis = true ∧ 
  C.eccentricity = 1/2 ∧ 
  C.passes_through = (1, 3/2) ∧
  l.point = (2, 1) →
  (∃ (A B : ℝ × ℝ), 
    -- C has equation x^2/4 + y^2/3 = 1
    (A.1^2/4 + A.2^2/3 = 1 ∧ B.1^2/4 + B.2^2/3 = 1) ∧
    -- A and B are on line l
    (A.2 - l.point.2 = l.slope * (A.1 - l.point.1) ∧ 
     B.2 - l.point.2 = l.slope * (B.1 - l.point.1)) ∧
    -- A and B are distinct
    A ≠ B ∧
    -- PA · PB = PM^2
    dot_product (A.1 - l.point.1, A.2 - l.point.2) (B.1 - l.point.1, B.2 - l.point.2) = 
    dot_product (1 - 2, 3/2 - 1) (1 - 2, 3/2 - 1) ∧
    -- l has equation y = (1/2)x
    l.slope = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2863_286300


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_seven_l2863_286347

theorem smallest_four_digit_mod_seven : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 7 = 6 → n ≥ 1000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_seven_l2863_286347


namespace NUMINAMATH_CALUDE_sons_ages_l2863_286331

theorem sons_ages (x y : ℕ+) (h1 : x < y) (h2 : y ≤ 4) 
  (h3 : ∃ (a b : ℕ+), a ≠ x ∧ b ≠ y ∧ a * b = x * y)
  (h4 : x ≠ y → (x = 1 ∧ y = 4)) :
  x = 1 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_sons_ages_l2863_286331


namespace NUMINAMATH_CALUDE_camp_kids_count_camp_kids_count_proof_l2863_286345

theorem camp_kids_count : ℕ → Prop :=
  fun total_kids =>
    let soccer_kids := total_kids / 2
    let morning_soccer_kids := soccer_kids / 4
    let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
    afternoon_soccer_kids = 750 ∧ total_kids = 2000

-- The proof goes here
theorem camp_kids_count_proof : ∃ n : ℕ, camp_kids_count n := by
  sorry

end NUMINAMATH_CALUDE_camp_kids_count_camp_kids_count_proof_l2863_286345


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2863_286398

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 3 / 5)
  (h2 : r / s = 5 / 4)
  (h3 : s / q = 1 / 3) :
  p / r = 36 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2863_286398


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l2863_286330

def is_prime (p : ℕ) : Prop := sorry

theorem largest_three_digit_product (n x y : ℕ) :
  n ≥ 100 ∧ n < 1000 ∧
  is_prime x ∧ is_prime y ∧ is_prime (10 * y - x) ∧
  x < 10 ∧ y < 10 ∧
  n = x * y * (10 * y - x) ∧
  x ≠ y ∧ x ≠ (10 * y - x) ∧ y ≠ (10 * y - x) →
  n ≤ 705 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l2863_286330


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2863_286369

theorem simple_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (t : ℝ)  -- Time period in years
  (SI : ℝ) -- Simple interest
  (h1 : P = 2800)
  (h2 : t = 5)
  (h3 : SI = P - 2240)
  (h4 : SI = (P * r * t) / 100) -- r is the interest rate
  : r = 4 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2863_286369


namespace NUMINAMATH_CALUDE_intersection_M_N_l2863_286387

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2863_286387


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l2863_286305

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lap_time : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photographer's setup -/
structure Photographer where
  start_time : ℕ  -- in seconds
  end_time : ℕ    -- in seconds
  track_coverage : ℚ  -- fraction of track covered in picture

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (linda : Runner) (luis : Runner) (photographer : Photographer) : ℚ :=
  sorry  -- Proof goes here

/-- The main theorem statement -/
theorem runners_in_picture_probability :
  let linda : Runner := { name := "Linda", lap_time := 120, direction := true }
  let luis : Runner := { name := "Luis", lap_time := 75, direction := false }
  let photographer : Photographer := { start_time := 900, end_time := 960, track_coverage := 1/3 }
  probability_both_in_picture linda luis photographer = 5/6 := by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_runners_in_picture_probability_l2863_286305


namespace NUMINAMATH_CALUDE_books_about_sports_l2863_286340

theorem books_about_sports (total_books school_books : ℕ) : 
  total_books = 58 → school_books = 19 → total_books - school_books = 39 := by
  sorry

end NUMINAMATH_CALUDE_books_about_sports_l2863_286340


namespace NUMINAMATH_CALUDE_solve_smores_problem_l2863_286354

def smores_problem (graham_crackers_per_smore : ℕ) 
                   (total_graham_crackers : ℕ) 
                   (initial_marshmallows : ℕ) 
                   (additional_marshmallows : ℕ) : Prop :=
  let total_smores := total_graham_crackers / graham_crackers_per_smore
  let total_marshmallows := initial_marshmallows + additional_marshmallows
  (total_marshmallows / total_smores = 1)

theorem solve_smores_problem :
  smores_problem 2 48 6 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_smores_problem_l2863_286354


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2863_286366

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ)^2 - ((t.base : ℚ) / 2)^2).sqrt) / 2

/-- Theorem: Minimum perimeter of two noncongruent isosceles triangles with same area and base ratio 9:8 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t1.base = 8 * t2.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      9 * s1.base = 8 * s2.base →
      perimeter t1 ≤ perimeter s1 :=
by sorry

#eval perimeter { leg := 90, base := 144 } -- Expected output: 324

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2863_286366


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2863_286338

theorem quadratic_minimum (x : ℝ) : x^2 - 4*x - 2019 ≥ -2023 := by
  sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, x^2 - 4*x - 2019 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2863_286338


namespace NUMINAMATH_CALUDE_difference_of_squares_l2863_286322

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 10) : x^2 - y^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2863_286322


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l2863_286372

theorem fraction_sum_integer (n : ℕ+) (h : ∃ (k : ℤ), (1/4 : ℚ) + (1/5 : ℚ) + (1/10 : ℚ) + (1/(n : ℚ)) = k) : n = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l2863_286372


namespace NUMINAMATH_CALUDE_cube_face_sum_theorem_l2863_286356

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex products for a NumberedCube -/
def vertexProductSum (cube : NumberedCube) : ℕ :=
  cube.a * cube.b * cube.c +
  cube.a * cube.e * cube.c +
  cube.a * cube.b * cube.f +
  cube.a * cube.e * cube.f +
  cube.d * cube.b * cube.c +
  cube.d * cube.e * cube.c +
  cube.d * cube.b * cube.f +
  cube.d * cube.e * cube.f

/-- Calculates the sum of face numbers for a NumberedCube -/
def faceSum (cube : NumberedCube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem: If the sum of vertex products is 357, then the sum of face numbers is 27 -/
theorem cube_face_sum_theorem (cube : NumberedCube) :
  vertexProductSum cube = 357 → faceSum cube = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_theorem_l2863_286356


namespace NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l2863_286365

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 12 * a - a^3

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l2863_286365


namespace NUMINAMATH_CALUDE_luke_stickers_l2863_286316

theorem luke_stickers (initial bought birthday given_away used : ℕ) :
  initial = 20 →
  bought = 12 →
  birthday = 20 →
  given_away = 5 →
  used = 8 →
  initial + bought + birthday - given_away - used = 39 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l2863_286316


namespace NUMINAMATH_CALUDE_x_coordinate_of_Q_l2863_286336

theorem x_coordinate_of_Q (P Q : ℝ × ℝ) (α : ℝ) : 
  P = (3/5, 4/5) →
  (Q.1 < 0 ∧ Q.2 < 0) →
  Real.sqrt (Q.1^2 + Q.2^2) = 1 →
  α = Real.arccos (3/5) →
  α + 3 * Real.pi / 4 = Real.arccos Q.1 →
  Q.1 = -7 * Real.sqrt 2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_x_coordinate_of_Q_l2863_286336


namespace NUMINAMATH_CALUDE_garden_area_l2863_286381

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) (longer_side_ratio : ℕ) : 
  total_posts = 20 →
  post_spacing = 4 →
  longer_side_ratio = 2 →
  ∃ (short_side long_side : ℕ),
    short_side * long_side = 336 ∧
    short_side * longer_side_ratio = long_side ∧
    short_side * post_spacing = (short_side - 1) * post_spacing ∧
    long_side * post_spacing = (long_side - 1) * post_spacing ∧
    2 * (short_side + long_side) - 4 = total_posts :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l2863_286381


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2863_286307

/-- A line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) : ℝ → ℝ := λ x ↦ m * (x - x₀) + y₀

theorem quadrilateral_area : 
  let line1 := Line (-3) 5 5
  let line2 := Line (-1) 10 0
  let B := (0, line1 0)
  let E := (5, 5)
  let C := (10, 0)
  (B.2 * C.1 - (B.2 * E.1 + C.1 * E.2)) / 2 = 125 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2863_286307


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_read_l2863_286335

/-- The number of books read in the 'Crazy Silly School' series -/
def books_read (total_books unread_books : ℕ) : ℕ :=
  total_books - unread_books

/-- Theorem stating that the number of books read is 33 -/
theorem crazy_silly_school_books_read :
  books_read 50 17 = 33 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_read_l2863_286335


namespace NUMINAMATH_CALUDE_book_cost_theorem_l2863_286399

/-- Calculates the cost of a single book given the total budget, remaining money, number of series bought, books per series, and tax rate. -/
def calculate_book_cost (total_budget : ℚ) (remaining_money : ℚ) (series_bought : ℕ) (books_per_series : ℕ) (tax_rate : ℚ) : ℚ :=
  let total_spent := total_budget - remaining_money
  let books_bought := series_bought * books_per_series
  let pre_tax_total := total_spent / (1 + tax_rate)
  let pre_tax_per_book := pre_tax_total / books_bought
  pre_tax_per_book * (1 + tax_rate)

/-- The cost of each book is approximately $5.96 given the problem conditions. -/
theorem book_cost_theorem :
  let total_budget : ℚ := 200
  let remaining_money : ℚ := 56
  let series_bought : ℕ := 3
  let books_per_series : ℕ := 8
  let tax_rate : ℚ := 1/10
  abs (calculate_book_cost total_budget remaining_money series_bought books_per_series tax_rate - 596/100) < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_book_cost_theorem_l2863_286399


namespace NUMINAMATH_CALUDE_distance_to_line_l2863_286385

/-- Given a triangle ABC with sides AB = 3, BC = 4, and CA = 5,
    the distance from point B to line AC is 12/5 -/
theorem distance_to_line (A B C : ℝ × ℝ) : 
  let d := (λ P Q : ℝ × ℝ => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  d A B = 3 ∧ d B C = 4 ∧ d C A = 5 → 
  (let area := (1/2) * d A B * d B C
   area / d C A) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l2863_286385


namespace NUMINAMATH_CALUDE_power_equality_l2863_286318

theorem power_equality (x : ℝ) (h : (2 : ℝ) ^ (3 * x) = 7) : (8 : ℝ) ^ (x + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2863_286318


namespace NUMINAMATH_CALUDE_class_size_problem_l2863_286310

theorem class_size_problem (passing_score : ℝ) (class_average : ℝ) 
  (pass_average_before : ℝ) (fail_average_before : ℝ)
  (pass_average_after : ℝ) (fail_average_after : ℝ)
  (points_added : ℝ) :
  passing_score = 65 →
  class_average = 66 →
  pass_average_before = 71 →
  fail_average_before = 56 →
  pass_average_after = 75 →
  fail_average_after = 59 →
  points_added = 5 →
  ∃ (total_students : ℕ), 
    15 < total_students ∧ 
    total_students < 30 ∧
    total_students = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l2863_286310


namespace NUMINAMATH_CALUDE_water_added_to_mixture_water_added_is_ten_l2863_286396

/-- Given a mixture of alcohol and water, prove the amount of water added to change the ratio. -/
theorem water_added_to_mixture (initial_ratio : ℚ) (final_ratio : ℚ) (alcohol_quantity : ℚ) : ℚ :=
  let initial_water := (alcohol_quantity * 5) / 2
  let water_added := (7 * alcohol_quantity) / 2 - initial_water
  by
    -- Assumptions
    have h1 : initial_ratio = 2 / 5 := by sorry
    have h2 : final_ratio = 2 / 7 := by sorry
    have h3 : alcohol_quantity = 10 := by sorry

    -- Proof
    sorry

/-- The amount of water added to the mixture is 10 liters. -/
theorem water_added_is_ten : water_added_to_mixture (2/5) (2/7) 10 = 10 := by sorry

end NUMINAMATH_CALUDE_water_added_to_mixture_water_added_is_ten_l2863_286396


namespace NUMINAMATH_CALUDE_max_a_for_decreasing_cos_minus_sin_l2863_286323

/-- The maximum value of a for which f(x) = cos x - sin x is decreasing on [-a, a] --/
theorem max_a_for_decreasing_cos_minus_sin (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, 
    ∀ y ∈ Set.Icc (-a) a, 
    x < y → (Real.cos x - Real.sin x) > (Real.cos y - Real.sin y)) → 
  a ≤ π/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_decreasing_cos_minus_sin_l2863_286323


namespace NUMINAMATH_CALUDE_pea_patch_part_size_l2863_286319

/-- Proves that the size of a part of a pea patch is 5 square feet -/
theorem pea_patch_part_size :
  ∀ (pea_patch radish_patch pea_part : ℝ),
  pea_patch = 2 * radish_patch →
  radish_patch = 15 →
  pea_part = pea_patch / 6 →
  pea_part = 5 := by
sorry

end NUMINAMATH_CALUDE_pea_patch_part_size_l2863_286319


namespace NUMINAMATH_CALUDE_max_value_of_z_l2863_286394

-- Define the objective function
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x y ≥ z x' y' ∧
  z x y = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2863_286394


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2863_286368

/-- Given an equation y = a + b/x, where a and b are constants, 
    prove that a - b = 19/2 when y = 2 for x = 2 and y = 7 for x = -2 -/
theorem a_minus_b_value (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = 2) ∧ (a + b / x = 7 ↔ x = -2)) → 
  a - b = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2863_286368


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2863_286332

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2863_286332


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l2863_286344

theorem opposite_numbers_equation (a b : ℝ) : a + b = 0 → a - (2 - b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l2863_286344


namespace NUMINAMATH_CALUDE_arc_length_120_degrees_l2863_286378

/-- Given a circle with circumference 90 meters and an arc subtended by a 120° central angle,
    prove that the length of the arc is 30 meters. -/
theorem arc_length_120_degrees (circle_circumference : ℝ) (central_angle : ℝ) (arc_length : ℝ) :
  circle_circumference = 90 →
  central_angle = 120 →
  arc_length = (central_angle / 360) * circle_circumference →
  arc_length = 30 := by
sorry

end NUMINAMATH_CALUDE_arc_length_120_degrees_l2863_286378


namespace NUMINAMATH_CALUDE_white_washing_cost_l2863_286343

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 10) :
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + num_windows * window_length * window_width)) * cost_per_sqft = 9060 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_l2863_286343


namespace NUMINAMATH_CALUDE_square_root_divided_by_three_l2863_286371

theorem square_root_divided_by_three : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_three_l2863_286371


namespace NUMINAMATH_CALUDE_pi_estimation_l2863_286397

theorem pi_estimation (n m : ℕ) (h1 : n = 100) (h2 : m = 31) :
  let π_est := 4 * (n : ℝ) / (m : ℝ) - 3
  π_est = 81 / 25 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l2863_286397


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2863_286359

theorem sum_mod_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2863_286359


namespace NUMINAMATH_CALUDE_cats_on_ship_l2863_286320

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 7

/-- Represents the number of humans on the ship -/
def num_humans : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

theorem cats_on_ship :
  (num_cats + num_humans = total_heads) ∧
  (4 * num_cats + 2 * num_humans - 1 = total_legs) :=
sorry

end NUMINAMATH_CALUDE_cats_on_ship_l2863_286320


namespace NUMINAMATH_CALUDE_list_number_fraction_l2863_286306

theorem list_number_fraction (list : List ℝ) (n : ℝ) : 
  list.length = 31 ∧ 
  n ∉ list ∧
  n = 5 * ((list.sum) / 30) →
  n / (list.sum + n) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_list_number_fraction_l2863_286306


namespace NUMINAMATH_CALUDE_trigonometric_properties_l2863_286395

theorem trigonometric_properties :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ∧
  ¬(∃ x : ℝ, Real.sin x ^ 2 + Real.cos x ^ 2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l2863_286395


namespace NUMINAMATH_CALUDE_group_size_l2863_286315

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_collection : ℕ := 1369

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = n

/-- The total collection is the product of the number of members and their contribution -/
axiom total_collection_eq : n * n = total_collection

theorem group_size : n = 37 := by sorry

end NUMINAMATH_CALUDE_group_size_l2863_286315


namespace NUMINAMATH_CALUDE_andrews_game_preparation_time_l2863_286386

/-- The time it takes to prepare all games -/
def total_preparation_time (time_per_game : ℕ) (num_games : ℕ) : ℕ :=
  time_per_game * num_games

/-- Theorem: The total preparation time for 5 games, each taking 5 minutes, is 25 minutes -/
theorem andrews_game_preparation_time :
  total_preparation_time 5 5 = 25 := by
sorry

end NUMINAMATH_CALUDE_andrews_game_preparation_time_l2863_286386


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2863_286339

def team_size : ℕ := 15
def lineup_size : ℕ := 7
def all_stars : ℕ := 3
def guards : ℕ := 5

theorem starting_lineup_count :
  (Finset.sum (Finset.range 3) (λ i =>
    Nat.choose guards (i + 2) * Nat.choose (team_size - all_stars - guards) (lineup_size - all_stars - (i + 2)))) = 285 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l2863_286339


namespace NUMINAMATH_CALUDE_ellipse_interfocal_distance_l2863_286324

/-- An ellipse with given latus rectum and focus-to-vertex distance has a specific interfocal distance -/
theorem ellipse_interfocal_distance 
  (latus_rectum : ℝ) 
  (focus_to_vertex : ℝ) 
  (h1 : latus_rectum = 5.4)
  (h2 : focus_to_vertex = 1.5) :
  ∃ (a b c : ℝ),
    a^2 = b^2 + c^2 ∧
    a - c = focus_to_vertex ∧
    b = latus_rectum / 2 ∧
    2 * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_interfocal_distance_l2863_286324


namespace NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l2863_286384

/-- Proves that in a kennel with a 2:3 ratio of cats to dogs and 18 dogs, there are 6 fewer cats than dogs -/
theorem kennel_cats_dogs_difference :
  ∀ (num_cats num_dogs : ℕ),
  num_dogs = 18 →
  num_cats * 3 = num_dogs * 2 →
  num_cats < num_dogs →
  num_dogs - num_cats = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l2863_286384


namespace NUMINAMATH_CALUDE_marly_soup_bags_l2863_286352

/-- The number of bags needed for Marly's soup -/
def bags_needed (milk_quarts chicken_stock_multiplier vegetable_quarts bag_capacity : ℚ) : ℚ :=
  (milk_quarts + chicken_stock_multiplier * milk_quarts + vegetable_quarts) / bag_capacity

/-- Theorem: Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_marly_soup_bags_l2863_286352


namespace NUMINAMATH_CALUDE_timmy_initial_money_l2863_286380

/-- Represents the properties of oranges and Timmy's situation --/
structure OrangeProblem where
  calories_per_orange : ℕ
  cost_per_orange : ℚ
  calories_needed : ℕ
  money_left : ℚ

/-- Calculates Timmy's initial amount of money --/
def initial_money (p : OrangeProblem) : ℚ :=
  let oranges_needed := p.calories_needed / p.calories_per_orange
  let oranges_cost := oranges_needed * p.cost_per_orange
  oranges_cost + p.money_left

/-- Theorem stating that given the problem conditions, Timmy's initial money was $10.00 --/
theorem timmy_initial_money :
  let p : OrangeProblem := {
    calories_per_orange := 80,
    cost_per_orange := 6/5, -- $1.20 represented as a rational number
    calories_needed := 400,
    money_left := 4
  }
  initial_money p = 10 := by sorry

end NUMINAMATH_CALUDE_timmy_initial_money_l2863_286380


namespace NUMINAMATH_CALUDE_angle_difference_range_l2863_286393

/-- Given an acute angle and the absolute difference between this angle and its supplementary angle -/
theorem angle_difference_range (x α : Real) : 
  (0 < x) → (x < 90) → (α = |180 - 2*x|) → (0 < α ∧ α < 180) := by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l2863_286393


namespace NUMINAMATH_CALUDE_triangle_area_on_grid_l2863_286303

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem triangle_area_on_grid :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 2, y := 0 }
  let C : Point := { x := 2, y := 2.5 }
  triangleArea A B C = 2.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_on_grid_l2863_286303


namespace NUMINAMATH_CALUDE_apple_price_theorem_l2863_286312

/-- The relationship between the selling price and quantity of apples -/
def apple_price_relation (x y : ℝ) : Prop :=
  y = 8 * x

/-- The price increase per kg of apples -/
def price_increase_per_kg : ℝ := 8

theorem apple_price_theorem (x y : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₂ - x₁ = 1 → apple_price_relation x₂ y₂ → apple_price_relation x₁ y₁ → y₂ - y₁ = price_increase_per_kg) →
  apple_price_relation x y :=
sorry

end NUMINAMATH_CALUDE_apple_price_theorem_l2863_286312


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2863_286361

theorem min_value_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2863_286361


namespace NUMINAMATH_CALUDE_machine_production_rate_l2863_286311

/-- The number of shirts a machine can make in one minute, given the total number of shirts and total time -/
def shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) : ℚ :=
  total_shirts / total_minutes

/-- Theorem stating that the machine makes 7 shirts per minute -/
theorem machine_production_rate :
  shirts_per_minute 196 28 = 7 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_rate_l2863_286311


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l2863_286326

/-- Calculates the overall profit percentage for cricket bat sales -/
theorem cricket_bat_profit_percentage
  (num_a : ℕ) (price_a : ℚ) (profit_a : ℚ)
  (num_b : ℕ) (price_b : ℚ) (profit_b : ℚ) :
  num_a = 5 ∧ price_a = 850 ∧ profit_a = 225 ∧
  num_b = 10 ∧ price_b = 950 ∧ profit_b = 300 →
  let total_profit := num_a * profit_a + num_b * profit_b
  let total_revenue := num_a * price_a + num_b * price_b
  (total_profit / total_revenue) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l2863_286326


namespace NUMINAMATH_CALUDE_exponent_multiplication_calculate_expression_l2863_286391

theorem exponent_multiplication (a : ℕ) (m n : ℕ) : 
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_expression : 3000 * (3000 ^ 2500) = 3000 ^ 2501 :=
by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_calculate_expression_l2863_286391


namespace NUMINAMATH_CALUDE_boys_in_basketball_camp_l2863_286353

theorem boys_in_basketball_camp (total : ℕ) (boy_ratio girl_ratio : ℕ) (boys girls : ℕ) : 
  total = 48 →
  boy_ratio = 3 →
  girl_ratio = 5 →
  boys + girls = total →
  boy_ratio * girls = girl_ratio * boys →
  boys = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_in_basketball_camp_l2863_286353


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2863_286350

/-- Given a point with rectangular coordinates (-5, -7, 4) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (5, 7, 4). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  (ρ * Real.sin φ * Real.cos θ = -5) →
  (ρ * Real.sin φ * Real.sin θ = -7) →
  (ρ * Real.cos φ = 4) →
  (ρ * Real.sin (-φ) * Real.cos (θ + π) = 5) ∧
  (ρ * Real.sin (-φ) * Real.sin (θ + π) = 7) ∧
  (ρ * Real.cos (-φ) = 4) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2863_286350


namespace NUMINAMATH_CALUDE_derivative_at_three_l2863_286379

theorem derivative_at_three : 
  let f (x : ℝ) := (x + 3) / (x^2 + 3)
  deriv f 3 = -1/6 := by sorry

end NUMINAMATH_CALUDE_derivative_at_three_l2863_286379


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l2863_286363

structure Space where
  Plane : Type
  Line : Type
  parallel : Plane → Plane → Prop
  perpendicular : Line → Plane → Prop
  subset : Line → Plane → Prop

theorem line_perpendicular_to_parallel_planes 
  (S : Space) (α β : S.Plane) (m : S.Line) : 
  S.perpendicular m α → S.parallel α β → S.perpendicular m β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l2863_286363


namespace NUMINAMATH_CALUDE_parallel_lines_angle_problem_l2863_286328

-- Define the angles as real numbers
variable (AXE CYX BXY : ℝ)

-- State the theorem
theorem parallel_lines_angle_problem 
  (h1 : AXE = 4 * CYX - 120) -- Given condition
  (h2 : AXE = CYX) -- From parallel lines property
  : BXY = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_problem_l2863_286328


namespace NUMINAMATH_CALUDE_dividing_line_ratio_l2863_286351

/-- A trapezoid with given dimensions and a dividing line -/
structure Trapezoid :=
  (base1 : ℝ)
  (base2 : ℝ)
  (leg1 : ℝ)
  (leg2 : ℝ)
  (dividing_ratio : ℝ × ℝ)

/-- The condition that the dividing line creates equal perimeters -/
def equal_perimeters (t : Trapezoid) : Prop :=
  let (m, n) := t.dividing_ratio
  let x := t.base1 + (t.base2 - t.base1) * (m / (m + n))
  t.base1 + m + x + t.leg1 * (m / (m + n)) =
  t.base2 + n + x + t.leg1 * (n / (m + n))

/-- The theorem stating the ratio of the dividing line -/
theorem dividing_line_ratio (t : Trapezoid) 
    (h1 : t.base1 = 3) 
    (h2 : t.base2 = 9) 
    (h3 : t.leg1 = 4) 
    (h4 : t.leg2 = 6) 
    (h5 : equal_perimeters t) : 
    t.dividing_ratio = (4, 1) := by
  sorry


end NUMINAMATH_CALUDE_dividing_line_ratio_l2863_286351


namespace NUMINAMATH_CALUDE_math_class_registration_l2863_286373

theorem math_class_registration (total : ℕ) (history : ℕ) (english : ℕ) (all_three : ℕ) (exactly_two : ℕ) :
  total = 68 →
  history = 21 →
  english = 34 →
  all_three = 3 →
  exactly_two = 7 →
  ∃ (math : ℕ), math = 14 ∧ 
    total = history + math + english - (exactly_two - all_three) - all_three :=
by sorry

end NUMINAMATH_CALUDE_math_class_registration_l2863_286373


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l2863_286392

theorem unique_factorial_sum (n : ℕ) : 2 * n * n.factorial + n.factorial = 2520 ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l2863_286392
