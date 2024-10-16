import Mathlib

namespace NUMINAMATH_CALUDE_largest_in_set_l286_28628

def S (a : ℝ) : Set ℝ := {-2*a, 3*a, 18/a, a^2, 2}

theorem largest_in_set :
  ∀ a : ℝ, a = 3 → 
  ∃ m : ℝ, m ∈ S a ∧ ∀ x ∈ S a, x ≤ m ∧ 
  m = 3*a ∧ m = a^2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l286_28628


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l286_28656

theorem maintenance_check_increase (original_time new_time : ℕ) 
  (h1 : original_time = 30) 
  (h2 : new_time = 60) : 
  (new_time - original_time) / original_time * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l286_28656


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l286_28695

theorem power_of_three_mod_eleven : 3^1320 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l286_28695


namespace NUMINAMATH_CALUDE_percentage_students_like_blue_l286_28670

/-- Proves that 30% of students like blue given the problem conditions --/
theorem percentage_students_like_blue :
  ∀ (total_students : ℕ) (blue_yellow_count : ℕ) (red_ratio : ℚ),
    total_students = 200 →
    blue_yellow_count = 144 →
    red_ratio = 2/5 →
    ∃ (blue_ratio : ℚ),
      blue_ratio = 3/10 ∧
      blue_ratio * total_students + 
      (1 - blue_ratio) * (1 - red_ratio) * total_students = blue_yellow_count :=
by sorry

end NUMINAMATH_CALUDE_percentage_students_like_blue_l286_28670


namespace NUMINAMATH_CALUDE_die_probability_l286_28649

theorem die_probability (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 11)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 4)
  (h4 : blue_faces = 2)
  (h5 : total_faces = red_faces + yellow_faces + blue_faces) :
  (yellow_faces : ℚ) / total_faces * (blue_faces : ℚ) / total_faces = 8 / 121 := by
  sorry

end NUMINAMATH_CALUDE_die_probability_l286_28649


namespace NUMINAMATH_CALUDE_least_integer_greater_than_two_plus_sqrt_three_squared_l286_28601

theorem least_integer_greater_than_two_plus_sqrt_three_squared :
  ∃ n : ℤ, (n = 14 ∧ (2 + Real.sqrt 3)^2 < n ∧ ∀ m : ℤ, (2 + Real.sqrt 3)^2 < m → n ≤ m) :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_two_plus_sqrt_three_squared_l286_28601


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l286_28674

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) (total_cost : ℚ) :
  num_dvds = 4 →
  cost_per_dvd = 6/5 →
  total_cost = num_dvds * cost_per_dvd →
  total_cost = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l286_28674


namespace NUMINAMATH_CALUDE_calculation_proof_l286_28623

theorem calculation_proof : (-8 - 1/3) - 12 - (-70) - (-8 - 1/3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l286_28623


namespace NUMINAMATH_CALUDE_star_equation_roots_l286_28679

-- Define the operation ※
def star (a b : ℝ) : ℝ := a^2 + a*b

-- Theorem statement
theorem star_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ star x 3 = -m) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_roots_l286_28679


namespace NUMINAMATH_CALUDE_sequence_properties_l286_28690

/-- Sequence b_n with sum of first n terms S_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def S : ℕ → ℝ := sorry

/-- Arithmetic sequence c_n -/
def c : ℕ → ℝ := sorry

/-- Sequence a_n formed by common terms of b_n and c_n in ascending order -/
def a : ℕ → ℝ := sorry

/-- The product of the first n terms of a_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, 2 * S n = 3 * (b n - 1)) ∧ 
  (c 1 = 5) ∧
  (c 1 + c 2 + c 3 = 27) →
  (∀ n : ℕ, b n = 3^n) ∧
  (∀ n : ℕ, c n = 4*n + 1) ∧
  (T 20 = 9^210) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l286_28690


namespace NUMINAMATH_CALUDE_boys_age_l286_28659

theorem boys_age (boy daughter wife father : ℕ) : 
  boy = 5 * daughter →
  wife = 5 * boy →
  father = 2 * wife →
  boy + daughter + wife + father = 81 →
  boy = 5 :=
by sorry

end NUMINAMATH_CALUDE_boys_age_l286_28659


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l286_28614

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → (b^2 + 3*b - 4 = 0) → (a^2 + 4*a + b - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l286_28614


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l286_28689

/-- The greatest distance between centers of two circles in a rectangle --/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 15)
  (h_diameter : circle_diameter = 7)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = Real.sqrt 185 ∧
    ∀ (d' : ℝ), d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l286_28689


namespace NUMINAMATH_CALUDE_cans_per_bag_l286_28633

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 63) (h2 : total_bags = 7) :
  total_cans / total_bags = 9 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l286_28633


namespace NUMINAMATH_CALUDE_fruit_store_discount_l286_28639

/-- 
Given a fruit store scenario with:
- Total weight of fruit: 1000kg
- Cost price: 7 yuan per kg
- Original selling price: 10 yuan per kg
- Half of the fruit is sold at original price
- Total profit must not be less than 2000 yuan

This theorem states that the minimum discount factor x for the remaining half of the fruit
satisfies: x ≤ 7/11
-/
theorem fruit_store_discount (total_weight : ℝ) (cost_price selling_price : ℝ) 
  (min_profit : ℝ) (x : ℝ) :
  total_weight = 1000 →
  cost_price = 7 →
  selling_price = 10 →
  min_profit = 2000 →
  (total_weight / 2 * (selling_price - cost_price) + 
   total_weight / 2 * (selling_price * (1 - x) - cost_price) ≥ min_profit) →
  x ≤ 7 / 11 := by
  sorry


end NUMINAMATH_CALUDE_fruit_store_discount_l286_28639


namespace NUMINAMATH_CALUDE_rope_art_fraction_l286_28692

theorem rope_art_fraction (total_rope : ℝ) (remaining_rope : ℝ) 
  (h1 : total_rope = 50)
  (h2 : remaining_rope = 20)
  (h3 : remaining_rope = (total_rope - total_rope * x) / 2)
  : x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_rope_art_fraction_l286_28692


namespace NUMINAMATH_CALUDE_class_size_proof_l286_28686

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ male : ℕ, total = 3 * male)
  (h3 : ∃ registered unregistered : ℕ, 
    registered + unregistered = total ∧ 
    registered = 3 * unregistered - 1) :
  total = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l286_28686


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l286_28648

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l286_28648


namespace NUMINAMATH_CALUDE_A_single_element_A_at_most_one_element_l286_28654

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - x + a + 2 = 0}

-- Theorem 1: A contains only one element iff a ∈ {0, -2+√5, -2-√5}
theorem A_single_element (a : ℝ) :
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = -2 + Real.sqrt 5 ∨ a = -2 - Real.sqrt 5 :=
sorry

-- Theorem 2: A contains at most one element iff a ∈ (-∞, -2-√5] ∪ {0} ∪ [-2+√5, +∞)
theorem A_at_most_one_element (a : ℝ) :
  (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔
  a ≤ -2 - Real.sqrt 5 ∨ a = 0 ∨ a ≥ -2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_A_single_element_A_at_most_one_element_l286_28654


namespace NUMINAMATH_CALUDE_lg_calculation_l286_28617

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem lg_calculation : (lg 2)^2 + lg 20 * lg 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l286_28617


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l286_28647

/-- The cost of one pen in rupees -/
def pen_cost : ℕ := 65

/-- The ratio of the cost of one pen to one pencil -/
def pen_pencil_ratio : ℚ := 5/1

/-- The cost of 3 pens and some pencils in rupees -/
def total_cost : ℕ := 260

/-- The number of pens in a dozen -/
def dozen : ℕ := 12

/-- Theorem stating that the cost of one dozen pens is 780 rupees -/
theorem cost_of_dozen_pens : pen_cost * dozen = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l286_28647


namespace NUMINAMATH_CALUDE_empty_box_weight_l286_28684

-- Define the number of balls
def num_balls : ℕ := 30

-- Define the weight of each ball in kg
def ball_weight : ℝ := 0.36

-- Define the total weight of the box with balls in kg
def total_weight : ℝ := 11.26

-- Theorem to prove
theorem empty_box_weight :
  total_weight - (num_balls : ℝ) * ball_weight = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_empty_box_weight_l286_28684


namespace NUMINAMATH_CALUDE_items_count_correct_l286_28694

/-- Given a number of children and the number of items each child has,
    calculate the total number of items for all children. -/
def totalItems (numChildren : ℕ) (itemsPerChild : ℕ) : ℕ :=
  numChildren * itemsPerChild

/-- Prove that for 12 children, each with 5 pencils, 3 erasers, 13 skittles, and 7 crayons,
    the total number of each item is correct. -/
theorem items_count_correct :
  let numChildren : ℕ := 12
  let pencilsPerChild : ℕ := 5
  let erasersPerChild : ℕ := 3
  let skittlesPerChild : ℕ := 13
  let crayonsPerChild : ℕ := 7
  
  (totalItems numChildren pencilsPerChild = 60) ∧
  (totalItems numChildren erasersPerChild = 36) ∧
  (totalItems numChildren skittlesPerChild = 156) ∧
  (totalItems numChildren crayonsPerChild = 84) :=
by sorry

end NUMINAMATH_CALUDE_items_count_correct_l286_28694


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l286_28624

-- Define the quadrilateral PQRS
def P (k a : ℤ) : ℤ × ℤ := (k, a)
def Q (k a : ℤ) : ℤ × ℤ := (a, k)
def R (k a : ℤ) : ℤ × ℤ := (-k, -a)
def S (k a : ℤ) : ℤ × ℤ := (-a, -k)

-- Define the area function for PQRS
def area_PQRS (k a : ℤ) : ℤ := 2 * |k - a| * |k + a|

-- Theorem statement
theorem quadrilateral_area_theorem (k a : ℤ) 
  (h1 : k > a) (h2 : a > 0) (h3 : area_PQRS k a = 32) : 
  k + a = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l286_28624


namespace NUMINAMATH_CALUDE_hilt_trip_distance_l286_28613

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : ℝ) : ℝ :=
  final_reading - initial_reading

theorem hilt_trip_distance :
  let initial_reading : ℝ := 212.3
  let final_reading : ℝ := 372
  distance_traveled initial_reading final_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_hilt_trip_distance_l286_28613


namespace NUMINAMATH_CALUDE_rebus_unique_solution_l286_28693

/-- A solution to the rebus system is a tuple of four distinct single-digit integers (M, A, H, P) -/
def RebusSolution := { t : Fin 10 × Fin 10 × Fin 10 × Fin 10 // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2.1 ∧ t.1 ≠ t.2.2.2 ∧ t.2.1 ≠ t.2.2.1 ∧ t.2.1 ≠ t.2.2.2 ∧ t.2.2.1 ≠ t.2.2.2 }

/-- The rebus system equations -/
def rebusEquations (s : RebusSolution) : Prop :=
  let M := s.val.1
  let A := s.val.2.1
  let H := s.val.2.2.1
  let P := s.val.2.2.2
  (M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P ∧
  (A * 10 + M) * (A * 10 + M) = P * 100 + H * 10 + M ∧
  M ≠ 0 ∧ P ≠ 0

/-- The unique solution to the rebus system -/
theorem rebus_unique_solution :
  ∃! s : RebusSolution, rebusEquations s ∧ s.val = (1, 3, 6, 9) :=
sorry

end NUMINAMATH_CALUDE_rebus_unique_solution_l286_28693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l286_28625

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 5 + a 7 = 10 →
  a 1 + a 10 = 9.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l286_28625


namespace NUMINAMATH_CALUDE_inequality_proof_l286_28612

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hab : a ≤ b) (hbc : b ≤ c) : 
  (a*x + b*y + c*z) * (x/a + y/b + z/c) ≤ (x+y+z)^2 * (a+c)^2 / (4*a*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l286_28612


namespace NUMINAMATH_CALUDE_bobs_head_start_l286_28600

/-- Proves that Bob's head-start is 1 mile given the conditions -/
theorem bobs_head_start (bob_speed jim_speed : ℝ) (catch_time : ℝ) (head_start : ℝ) : 
  bob_speed = 6 → 
  jim_speed = 9 → 
  catch_time = 20 / 60 →
  head_start + bob_speed * catch_time = jim_speed * catch_time →
  head_start = 1 := by
sorry

end NUMINAMATH_CALUDE_bobs_head_start_l286_28600


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l286_28622

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the intersection point P
def P : Point := (2, 1)

-- Define the lines from the problem
def line1 : Line := λ x y ↦ 2*x + y - 5
def line2 : Line := λ x y ↦ x - 2*y
def line_l1 : Line := λ x y ↦ 4*x - y + 1

-- Define a function to check if a point is on a line
def on_line (p : Point) (l : Line) : Prop :=
  l p.1 p.2 = 0

-- Define parallel and perpendicular relationships
def parallel (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 x y

def perpendicular (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y * l2 x y = -k * (l1 y x * l2 y x)

-- State the theorems
theorem parallel_line_equation :
  ∀ l : Line, on_line P l ∧ parallel l line_l1 →
  ∀ x y, l x y = 4*x - y - 7 := by sorry

theorem perpendicular_line_equation :
  ∀ l : Line, on_line P l ∧ perpendicular l line_l1 →
  ∀ x y, l x y = x + 4*y - 6 := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l286_28622


namespace NUMINAMATH_CALUDE_trig_identity_proof_l286_28634

theorem trig_identity_proof (α : Real) (h : Real.tan α = 3) : 
  (Real.cos (α + π/4))^2 - (Real.cos (α - π/4))^2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l286_28634


namespace NUMINAMATH_CALUDE_total_notes_count_l286_28616

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes, 
    with 57 notes of Rs. 50 denomination, the total number of notes is 72. -/
theorem total_notes_count (total_amount : ℕ) (fifty_note_count : ℕ) : 
  total_amount = 10350 →
  fifty_note_count = 57 →
  ∃ (five_hundred_note_count : ℕ),
    total_amount = fifty_note_count * 50 + five_hundred_note_count * 500 ∧
    fifty_note_count + five_hundred_note_count = 72 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l286_28616


namespace NUMINAMATH_CALUDE_circle_equation_l286_28661

theorem circle_equation (x y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 + p.2^2 = 16)) ∧
    ((-4, 0) ∈ C) ∧
    ((x, y) ∈ C)) →
  x^2 + y^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l286_28661


namespace NUMINAMATH_CALUDE_number_in_mind_l286_28652

theorem number_in_mind (x : ℝ) : (x - 6) / 13 = 2 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_in_mind_l286_28652


namespace NUMINAMATH_CALUDE_line_equation_l286_28631

/-- The distance between intersection points of x = k with y = x^2 + 4x + 4 and y = mx + b is 10 -/
def intersection_distance (m b k : ℝ) : Prop :=
  |k^2 + 4*k + 4 - (m*k + b)| = 10

/-- The line y = mx + b passes through the point (1, 6) -/
def passes_through_point (m b : ℝ) : Prop :=
  m * 1 + b = 6

theorem line_equation (m b : ℝ) (h1 : ∃ k, intersection_distance m b k)
    (h2 : passes_through_point m b) (h3 : b ≠ 0) :
    m = 4 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l286_28631


namespace NUMINAMATH_CALUDE_raj_earnings_l286_28676

/-- Calculates the total earnings for Raj over two weeks given the hours worked and wage difference --/
def total_earnings (hours_week1 hours_week2 : ℕ) (wage_difference : ℚ) : ℚ :=
  let hourly_wage := wage_difference / (hours_week2 - hours_week1)
  (hours_week1 + hours_week2) * hourly_wage

/-- Proves that Raj's total earnings for the first two weeks of July is $198.00 --/
theorem raj_earnings :
  let hours_week1 : ℕ := 12
  let hours_week2 : ℕ := 18
  let wage_difference : ℚ := 39.6
  total_earnings hours_week1 hours_week2 wage_difference = 198 := by
  sorry

#eval total_earnings 12 18 (39.6 : ℚ)

end NUMINAMATH_CALUDE_raj_earnings_l286_28676


namespace NUMINAMATH_CALUDE_expand_product_l286_28691

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l286_28691


namespace NUMINAMATH_CALUDE_tiling_combination_l286_28620

def interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

def can_tile (a b c : ℕ) : Prop :=
  ∃ (m n p : ℕ), m * interior_angle a + n * interior_angle b + p * interior_angle c = 360 ∧
  m + n + p = 4 ∧ m > 0 ∧ n > 0 ∧ p > 0

theorem tiling_combination :
  can_tile 3 4 6 ∧
  ¬can_tile 3 4 5 ∧
  ¬can_tile 3 4 7 ∧
  ¬can_tile 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_tiling_combination_l286_28620


namespace NUMINAMATH_CALUDE_min_cars_in_group_l286_28637

/-- Represents the group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ

/-- The conditions of the car group -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 49 ∧
  g.withStripes ≥ 51 ∧
  g.withAC - g.withStripes ≤ 49

/-- The theorem stating that the minimum number of cars in a valid group is 100 -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 100 := by
  sorry

#check min_cars_in_group

end NUMINAMATH_CALUDE_min_cars_in_group_l286_28637


namespace NUMINAMATH_CALUDE_odd_function_a_value_l286_28682

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x =>
  if x > 0 then 1 + a^x else -1 - a^(-x)

-- State the theorem
theorem odd_function_a_value :
  ∀ a : ℝ,
  a > 0 →
  a ≠ 1 →
  (∀ x : ℝ, f a (-x) = -(f a x)) →
  f a (-1) = -3/2 →
  a = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l286_28682


namespace NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l286_28653

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo (-2 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l286_28653


namespace NUMINAMATH_CALUDE_project_completion_time_l286_28642

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The total number of days the project takes when A and B work together, with A quitting 15 days before completion -/
def total_days : ℝ := 21

/-- The number of days before project completion that A quits -/
def A_quit_days : ℝ := 15

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

theorem project_completion_time :
  A_days = 20 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l286_28642


namespace NUMINAMATH_CALUDE_at_least_two_correct_coats_l286_28644

theorem at_least_two_correct_coats (n : ℕ) (h : n = 5) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => (n.choose (k + 2)) * ((n - k - 2).factorial))) = 31 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_correct_coats_l286_28644


namespace NUMINAMATH_CALUDE_g_ln_inverse_2017_l286_28672

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

axiom a_positive : a > 0
axiom a_not_one : a ≠ 1
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom g_ln_2017 : g (Real.log 2017) = 2018

theorem g_ln_inverse_2017 : g (Real.log (1 / 2017)) = -2015 := by
  sorry

end NUMINAMATH_CALUDE_g_ln_inverse_2017_l286_28672


namespace NUMINAMATH_CALUDE_max_value_a_l286_28641

theorem max_value_a (a b c d : ℤ) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 4 * d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℤ), a' = 2367 ∧ a' < 2 * b' ∧ b' < 3 * c' ∧ c' < 4 * d' ∧ d' < 100 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l286_28641


namespace NUMINAMATH_CALUDE_calculate_ambulance_ride_cost_l286_28698

/-- Given a hospital bill with various components, calculate the cost of the ambulance ride. -/
theorem calculate_ambulance_ride_cost (total_bill : ℝ) (medication_percent : ℝ) 
  (imaging_percent : ℝ) (surgical_percent : ℝ) (overnight_percent : ℝ) (doctor_percent : ℝ) 
  (food_fee : ℝ) (consultation_fee : ℝ) (therapy_fee : ℝ) 
  (h1 : total_bill = 18000)
  (h2 : medication_percent = 35)
  (h3 : imaging_percent = 15)
  (h4 : surgical_percent = 25)
  (h5 : overnight_percent = 10)
  (h6 : doctor_percent = 5)
  (h7 : food_fee = 300)
  (h8 : consultation_fee = 450)
  (h9 : therapy_fee = 600) :
  total_bill - (medication_percent / 100 * total_bill + 
                imaging_percent / 100 * total_bill + 
                surgical_percent / 100 * total_bill + 
                overnight_percent / 100 * total_bill + 
                doctor_percent / 100 * total_bill + 
                food_fee + consultation_fee + therapy_fee) = 450 := by
  sorry


end NUMINAMATH_CALUDE_calculate_ambulance_ride_cost_l286_28698


namespace NUMINAMATH_CALUDE_f_is_even_l286_28683

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_is_even (g : ℝ → ℝ) (h : isEven g) :
  isEven (fun x ↦ |g (x^3)|) := by sorry

end NUMINAMATH_CALUDE_f_is_even_l286_28683


namespace NUMINAMATH_CALUDE_production_growth_rate_l286_28662

theorem production_growth_rate (initial_volume : ℝ) (final_volume : ℝ) (years : ℕ) (growth_rate : ℝ) : 
  initial_volume = 1000000 → 
  final_volume = 1210000 → 
  years = 2 →
  initial_volume * (1 + growth_rate) ^ years = final_volume →
  growth_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_production_growth_rate_l286_28662


namespace NUMINAMATH_CALUDE_club_officer_selection_l286_28657

theorem club_officer_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n * (n - 1) * (n - 2) = 6840 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l286_28657


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l286_28608

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_ten_million (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10000000

theorem smallest_divisible_term : 
  (∀ n < 8, ¬ is_divisible_by_ten_million (geometric_sequence (5/6) 25 n)) ∧ 
  is_divisible_by_ten_million (geometric_sequence (5/6) 25 8) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l286_28608


namespace NUMINAMATH_CALUDE_factorization_equality_l286_28655

theorem factorization_equality (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l286_28655


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l286_28606

def A (a : ℝ) : Set ℝ := {2, 3, a^2 + 4*a + 2}

def B (a : ℝ) : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}

theorem set_intersection_and_union (a : ℝ) :
  A a ∩ B a = {3, 7} → a = 1 ∧ A a ∪ B a = {0, 1, 2, 3, 7} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l286_28606


namespace NUMINAMATH_CALUDE_price_of_short_is_13_50_l286_28687

/-- The price of a single short, given the conditions of Jimmy and Irene's shopping trip -/
def price_of_short (num_shorts : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) 
  (discount_rate : ℚ) (total_paid : ℚ) : ℚ :=
  let shirt_total := num_shirts * shirt_price
  let discounted_shirt_total := shirt_total * (1 - discount_rate)
  let shorts_total := total_paid - discounted_shirt_total
  shorts_total / num_shorts

/-- Theorem stating that the price of each short is $13.50 under the given conditions -/
theorem price_of_short_is_13_50 :
  price_of_short 3 5 17 (1/10) 117 = 27/2 := by sorry

end NUMINAMATH_CALUDE_price_of_short_is_13_50_l286_28687


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l286_28615

theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1 ∧ a ≠ -2) :
  (a^2 - 3*a + 2) / (a^2 + a - 2) = (a - 2) / (a + 2) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_l286_28615


namespace NUMINAMATH_CALUDE_postcard_probability_l286_28675

/-- The probability of arranging n unique items in a line, such that k specific items are consecutive. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 unique postcards in a line, 
    such that 4 specific postcards are consecutive, is 1/55. -/
theorem postcard_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_postcard_probability_l286_28675


namespace NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_derivative_of_periodic_is_periodic_l286_28605

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The statement that the derivative of a periodic function is also periodic -/
theorem derivative_of_periodic_is_periodic (f : ℝ → ℝ) (T : ℝ) (hf : Differentiable ℝ f) 
    (h_periodic : IsPeriodic f T) : IsPeriodic (deriv f) T := by
  sorry

/-- Alternative formulation using a more explicit definition of periodicity -/
theorem derivative_of_periodic_is_periodic' (f : ℝ → ℝ) (T : ℝ) (hf : Differentiable ℝ f) 
    (h_periodic : ∀ x, f (x + T) = f x) : ∀ x, deriv f (x + T) = deriv f x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_derivative_of_periodic_is_periodic_l286_28605


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l286_28660

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 6

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l286_28660


namespace NUMINAMATH_CALUDE_railroad_cars_theorem_l286_28665

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Minimum number of tries needed to determine if there is an equal number of both types of cars -/
def minTries (totalCars : ℕ) : ℕ := totalCars - sumBinaryDigits totalCars

theorem railroad_cars_theorem :
  let totalCars : ℕ := 2022
  minTries totalCars = 2014 := by sorry

end NUMINAMATH_CALUDE_railroad_cars_theorem_l286_28665


namespace NUMINAMATH_CALUDE_time_saved_two_pipes_l286_28699

/-- Represents the time saved when using two pipes instead of one to fill a reservoir -/
theorem time_saved_two_pipes (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  let time_saved := p - (a * p) / (a + b)
  time_saved = (b * p) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_time_saved_two_pipes_l286_28699


namespace NUMINAMATH_CALUDE_product_in_base7_l286_28663

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base 7 number to base 10 --/
def fromBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem product_in_base7 : 
  multiplyBase7 (toBase7 231) (toBase7 452) = 613260 := by sorry

end NUMINAMATH_CALUDE_product_in_base7_l286_28663


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l286_28643

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) 
  (h_geom_seq : (4 : ℝ) * 9 = m^2) :
  let e := if m > 0 
    then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
    else Real.sqrt (1 + 6 / m) / 1
  e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l286_28643


namespace NUMINAMATH_CALUDE_fifth_power_sum_l286_28697

theorem fifth_power_sum (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 123 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l286_28697


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l286_28651

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem 
  (A : ℝ × ℝ) -- Point A
  (h1 : parabola A.1 A.2) -- A is on the parabola
  (h2 : ‖A - focus‖ = ‖B - focus‖) -- |AF| = |BF|
  : ‖A - B‖ = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l286_28651


namespace NUMINAMATH_CALUDE_smallest_valid_number_l286_28664

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  ∃ (k : Nat), 
    (n / (n % 100) = k^2) ∧
    (k^2 = (n / 100 + 1)^2)

theorem smallest_valid_number : 
  is_valid_number 1805 ∧ 
  ∀ (m : Nat), is_valid_number m → m ≥ 1805 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l286_28664


namespace NUMINAMATH_CALUDE_order_of_rational_numbers_l286_28645

theorem order_of_rational_numbers
  (a b c d : ℚ)
  (sum_eq : a + b = c + d)
  (ineq_1 : a + d < b + c)
  (ineq_2 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end NUMINAMATH_CALUDE_order_of_rational_numbers_l286_28645


namespace NUMINAMATH_CALUDE_real_return_calculation_l286_28635

theorem real_return_calculation (nominal_rate inflation_rate : ℝ) 
  (h1 : nominal_rate = 0.21)
  (h2 : inflation_rate = 0.10) :
  (1 + nominal_rate) / (1 + inflation_rate) - 1 = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_real_return_calculation_l286_28635


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l286_28632

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l286_28632


namespace NUMINAMATH_CALUDE_probability_heart_king_king_ace_l286_28671

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts excluding King and Ace of hearts -/
def HeartsExcludingKingAce : ℕ := 11

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of Aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Probability of drawing the specific sequence (Heart, King, King, Ace) -/
def probabilityHeartKingKingAce : ℚ :=
  (HeartsExcludingKingAce : ℚ) / StandardDeck *
  KingsInDeck / (StandardDeck - 1) *
  (KingsInDeck - 1) / (StandardDeck - 2) *
  AcesInDeck / (StandardDeck - 3)

theorem probability_heart_king_king_ace :
  probabilityHeartKingKingAce = 1 / 12317 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_king_king_ace_l286_28671


namespace NUMINAMATH_CALUDE_line_mb_product_l286_28667

/-- Given a line y = mx + b passing through points (0, -3) and (2, 3), prove that mb = -9 -/
theorem line_mb_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (3 : ℝ) = m * 2 + b → -- The line passes through (2, 3)
  m * b = -9 := by
sorry

end NUMINAMATH_CALUDE_line_mb_product_l286_28667


namespace NUMINAMATH_CALUDE_gcd_324_243_l286_28650

theorem gcd_324_243 : Nat.gcd 324 243 = 81 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_243_l286_28650


namespace NUMINAMATH_CALUDE_find_other_number_l286_28640

theorem find_other_number (n m : ℕ+) 
  (h_lcm : Nat.lcm n m = 52)
  (h_gcd : Nat.gcd n m = 8)
  (h_n : n = 26) : 
  m = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l286_28640


namespace NUMINAMATH_CALUDE_negative_integral_of_negative_function_l286_28621

theorem negative_integral_of_negative_function 
  {f : ℝ → ℝ} {a b : ℝ} 
  (hf : Continuous f) 
  (hneg : ∀ x, f x < 0) 
  (hab : a < b) : 
  ∫ x in a..b, f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_integral_of_negative_function_l286_28621


namespace NUMINAMATH_CALUDE_meaningful_zero_power_l286_28603

theorem meaningful_zero_power (m : ℝ) (h : m ≠ -1) : (m + 1) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_zero_power_l286_28603


namespace NUMINAMATH_CALUDE_tenth_letter_shift_l286_28618

def shift_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_letter_shift :
  ∀ (letter : Char),
  (shift_sum 10) % 26 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_letter_shift_l286_28618


namespace NUMINAMATH_CALUDE_inverse_function_sum_l286_28607

/-- Given a function g and constants a, b, c, d, k satisfying certain conditions,
    prove that a + d = 0 -/
theorem inverse_function_sum (a b c d k : ℝ) :
  (∀ x, (k * (a * x + b)) / (k * (c * x + d)) = 
        ((k * (a * ((k * (a * x + b)) / (k * (c * x + d))) + b)) / 
         (k * (c * ((k * (a * x + b)) / (k * (c * x + d))) + d)))) →
  (a * b * c * d * k ≠ 0) →
  (a + k * c = 0) →
  (a + d = 0) := by
sorry


end NUMINAMATH_CALUDE_inverse_function_sum_l286_28607


namespace NUMINAMATH_CALUDE_norris_savings_l286_28677

/-- The amount of money Norris saved in November -/
def november_savings : ℤ := sorry

/-- The amount of money Norris saved in September -/
def september_savings : ℤ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℤ := 25

/-- The amount of money Norris spent on an online game -/
def online_game_cost : ℤ := 75

/-- The amount of money Norris has left -/
def money_left : ℤ := 10

theorem norris_savings : november_savings = 31 := by
  sorry

end NUMINAMATH_CALUDE_norris_savings_l286_28677


namespace NUMINAMATH_CALUDE_final_position_of_E_l286_28680

-- Define the position of E as a pair of axes (base_axis, top_axis)
inductive Axis
  | PositiveX
  | NegativeX
  | PositiveY
  | NegativeY

def Position := Axis × Axis

-- Define the transformations
def rotateClockwise270 (p : Position) : Position :=
  match p with
  | (Axis.NegativeX, Axis.PositiveY) => (Axis.PositiveY, Axis.NegativeX)
  | _ => p  -- For completeness, though we only care about the initial position

def reflectXAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | _ => base,
      top
    )

def reflectYAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      base,
      match top with
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
      | _ => top
    )

def halfTurn (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX,
      match top with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
    )

-- Theorem statement
theorem final_position_of_E :
  let initial_position : Position := (Axis.NegativeX, Axis.PositiveY)
  let final_position := halfTurn (reflectYAxis (reflectXAxis (rotateClockwise270 initial_position)))
  final_position = (Axis.NegativeY, Axis.NegativeX) :=
by
  sorry

end NUMINAMATH_CALUDE_final_position_of_E_l286_28680


namespace NUMINAMATH_CALUDE_intersection_M_N_l286_28636

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l286_28636


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l286_28627

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ a < -1 ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l286_28627


namespace NUMINAMATH_CALUDE_inequality_proof_l286_28604

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l286_28604


namespace NUMINAMATH_CALUDE_investment_difference_l286_28685

def initial_investment : ℝ := 500

def jackson_multiplier : ℝ := 4

def brandon_percentage : ℝ := 0.2

def jackson_final (initial : ℝ) (multiplier : ℝ) : ℝ := initial * multiplier

def brandon_final (initial : ℝ) (percentage : ℝ) : ℝ := initial * percentage

theorem investment_difference :
  jackson_final initial_investment jackson_multiplier - brandon_final initial_investment brandon_percentage = 1900 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l286_28685


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l286_28696

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 258 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 86)
  (eq3 : a * d + b * c = 180)
  (eq4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 258 ∧ ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 258 ∧ 
    a + b = 17 ∧ a * b + c + d = 86 ∧ a * d + b * c = 180 ∧ c * d = 110 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l286_28696


namespace NUMINAMATH_CALUDE_binary_product_l286_28630

-- Define the binary numbers
def binary1 : Nat := 0b11011
def binary2 : Nat := 0b111
def binary3 : Nat := 0b101

-- Define the result
def result : Nat := 0b1110110001

-- Theorem statement
theorem binary_product :
  binary1 * binary2 * binary3 = result := by
  sorry

end NUMINAMATH_CALUDE_binary_product_l286_28630


namespace NUMINAMATH_CALUDE_field_trip_attendance_l286_28609

theorem field_trip_attendance (vans buses : ℕ) (people_per_van people_per_bus : ℕ) 
  (h1 : vans = 9)
  (h2 : buses = 10)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 27) :
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l286_28609


namespace NUMINAMATH_CALUDE_trig_simplification_l286_28646

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) =
  Real.tan (35 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l286_28646


namespace NUMINAMATH_CALUDE_friends_bread_slices_l286_28688

/-- Calculates the number of slices each friend eats given the number of friends and the slices in each loaf -/
def slices_per_friend (n : ℕ) (loaf1 loaf2 loaf3 loaf4 : ℕ) : ℕ :=
  (loaf1 + loaf2 + loaf3 + loaf4)

/-- Theorem stating that each friend eats 78 slices of bread -/
theorem friends_bread_slices (n : ℕ) (h : n > 0) :
  slices_per_friend n 15 18 20 25 = 78 := by
  sorry

#check friends_bread_slices

end NUMINAMATH_CALUDE_friends_bread_slices_l286_28688


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l286_28619

theorem point_on_terminal_side (t : ℝ) (θ : ℝ) : 
  ((-2 : ℝ) = Real.cos θ * Real.sqrt (4 + t^2)) →
  (t = Real.sin θ * Real.sqrt (4 + t^2)) →
  (Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) →
  t = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l286_28619


namespace NUMINAMATH_CALUDE_uncoverable_iff_odd_specified_boards_uncoverable_l286_28629

/-- Represents a board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (missing : ℕ)

/-- Calculates the number of coverable squares on a board -/
def coverableSquares (b : Board) : ℕ :=
  b.rows * b.cols - b.missing

/-- Determines if a board can be completely covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  coverableSquares b % 2 = 0

/-- Theorem: A board cannot be covered iff the number of coverable squares is odd -/
theorem uncoverable_iff_odd (b : Board) :
  ¬(canBeCovered b) ↔ coverableSquares b % 2 = 1 :=
sorry

/-- Examples of board configurations -/
def board_7x3 : Board := ⟨7, 3, 0⟩
def board_6x4_unpainted : Board := ⟨6, 4, 1⟩
def board_5x7 : Board := ⟨5, 7, 0⟩
def board_8x8_missing : Board := ⟨8, 8, 1⟩

/-- Theorem: The specified boards cannot be covered -/
theorem specified_boards_uncoverable :
  (¬(canBeCovered board_7x3)) ∧
  (¬(canBeCovered board_6x4_unpainted)) ∧
  (¬(canBeCovered board_5x7)) ∧
  (¬(canBeCovered board_8x8_missing)) :=
sorry

end NUMINAMATH_CALUDE_uncoverable_iff_odd_specified_boards_uncoverable_l286_28629


namespace NUMINAMATH_CALUDE_solve_equation_l286_28658

theorem solve_equation (n : ℚ) : 
  (1/(n+2)) + (3/(n+2)) + (2*n/(n+2)) = 4 → n = -2 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l286_28658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l286_28626

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 9 = 16 → a 5 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l286_28626


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l286_28611

/-- The value of a for which the tangents to C₁ and C₂ at their intersection point are perpendicular -/
theorem perpendicular_tangents_intersection (a : ℝ) : 
  a > 0 → 
  ∃ (x y : ℝ), 
    (y = a * x^3 + 1) ∧ 
    (x^2 + y^2 = 5/2) ∧ 
    (∃ (m₁ m₂ : ℝ), 
      (m₁ = 3 * a * x^2) ∧ 
      (m₂ = -x / y) ∧ 
      (m₁ * m₂ = -1)) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l286_28611


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l286_28666

/-- A regular tetrahedron with specific properties -/
structure RegularTetrahedron where
  -- The distance from the midpoint of the height to a lateral face
  midpoint_to_face : ℝ
  -- The distance from the midpoint of the height to a lateral edge
  midpoint_to_edge : ℝ

/-- The volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of a specific regular tetrahedron -/
theorem volume_of_specific_tetrahedron :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l286_28666


namespace NUMINAMATH_CALUDE_multiples_properties_l286_28602

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 6 * m) : 
  (∃ n : ℤ, b = 3 * n) ∧ 
  (∃ p : ℤ, a - b = 3 * p) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l286_28602


namespace NUMINAMATH_CALUDE_frame_area_percentage_l286_28638

theorem frame_area_percentage (square_side : ℝ) (frame_width : ℝ) : 
  square_side = 80 → frame_width = 4 → 
  (square_side^2 - (square_side - 2 * frame_width)^2) / square_side^2 * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_frame_area_percentage_l286_28638


namespace NUMINAMATH_CALUDE_new_person_age_l286_28673

/-- Given a group of 10 persons, prove that if replacing a 40-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 10 years. -/
theorem new_person_age (T : ℕ) (A : ℕ) : 
  (T / 10 : ℚ) - ((T - 40 + A) / 10 : ℚ) = 3 → A = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l286_28673


namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_zero_l286_28678

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => modifiedFibonacci (n + 1) + modifiedFibonacci n

def unitsDigit (n : ℕ) : ℕ := n % 10

def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, k ≤ n ∧ unitsDigit (modifiedFibonacci k) = d

theorem last_digit_to_appear_is_zero :
  ∃ N : ℕ, allDigitsAppeared N ∧
    ¬(allDigitsAppeared (N - 1)) ∧
    unitsDigit (modifiedFibonacci N) = 0 :=
  sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_zero_l286_28678


namespace NUMINAMATH_CALUDE_probability_not_blue_l286_28668

def odds_blue : ℚ := 5 / 6

theorem probability_not_blue (odds : ℚ) (h : odds = odds_blue) :
  1 - odds / (1 + odds) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_blue_l286_28668


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l286_28681

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (6, x)
  parallel a b → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l286_28681


namespace NUMINAMATH_CALUDE_constant_term_implies_a_l286_28610

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (ax^2 + 1/√x)^5 -/
def constantTerm (a : ℝ) : ℝ := a * (binomial 5 4)

theorem constant_term_implies_a (a : ℝ) :
  constantTerm a = -10 → a = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_implies_a_l286_28610


namespace NUMINAMATH_CALUDE_one_minus_repeating_thirds_l286_28669

/-- The decimal 0.333... (repeating 3) -/
def repeating_thirds : ℚ :=
  1 / 3

theorem one_minus_repeating_thirds :
  1 - repeating_thirds = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_thirds_l286_28669
