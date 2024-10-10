import Mathlib

namespace solution_difference_l1733_173354

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 17 * p - 68 →
  (q - 4) * (q + 4) = 17 * q - 68 →
  p ≠ q →
  p > q →
  p - q = 9 := by sorry

end solution_difference_l1733_173354


namespace gerald_pie_purchase_l1733_173337

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

/-- The cost of the meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def remaining_pfennigs : ℕ := 7

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

theorem gerald_pie_purchase :
  initial_farthings - pie_cost * farthings_per_pfennig = remaining_pfennigs * farthings_per_pfennig :=
sorry

end gerald_pie_purchase_l1733_173337


namespace intersection_of_A_and_complement_of_B_l1733_173347

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_complement_of_B : A ∩ (U \ B) = {-1} := by
  sorry

end intersection_of_A_and_complement_of_B_l1733_173347


namespace ribbon_fraction_per_gift_l1733_173372

theorem ribbon_fraction_per_gift 
  (total_fraction : ℚ) 
  (num_gifts : ℕ) 
  (h1 : total_fraction = 4 / 15) 
  (h2 : num_gifts = 5) : 
  total_fraction / num_gifts = 4 / 75 := by
  sorry

end ribbon_fraction_per_gift_l1733_173372


namespace circle_radius_with_min_distance_to_line_l1733_173366

/-- The radius of a circle with center (3, -5) that has a minimum distance of 1 to the line 4x - 3y - 2 = 0 -/
theorem circle_radius_with_min_distance_to_line : ∃ (r : ℝ), 
  r > 0 ∧ 
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
    ∃ (d : ℝ), d ≥ 1 ∧ d = |4*x - 3*y - 2| / (5 : ℝ)) ∧
  r = 4 :=
sorry

end circle_radius_with_min_distance_to_line_l1733_173366


namespace profit_maximizing_price_l1733_173319

/-- The profit function based on price increase -/
def profit (x : ℝ) : ℝ := (90 + x - 80) * (400 - 10 * x)

/-- The initial purchase price -/
def initial_purchase_price : ℝ := 80

/-- The initial selling price -/
def initial_selling_price : ℝ := 90

/-- The initial sales volume -/
def initial_sales_volume : ℝ := 400

/-- The rate of decrease in sales volume per unit price increase -/
def sales_decrease_rate : ℝ := 10

/-- Theorem stating that the profit-maximizing selling price is 105 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 15 ∧ 
  ∀ (y : ℝ), profit y ≤ profit x ∧
  initial_selling_price + x = 105 := by
  sorry

end profit_maximizing_price_l1733_173319


namespace distribute_eq_choose_l1733_173357

/-- The number of ways to distribute n items into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_eq_choose (n k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  distribute n k = choose (n - 1) (k - 1) :=
sorry

end distribute_eq_choose_l1733_173357


namespace sheep_problem_l1733_173359

theorem sheep_problem (n : ℕ) (h1 : n > 0) :
  let total := n * n
  let remainder := total % 10
  let elder_share := total - remainder
  let younger_share := remainder
  (remainder < 10 ∧ elder_share % 20 = 10) →
  (elder_share + younger_share + 2) / 2 = (elder_share + 2) / 2 ∧
  (elder_share + younger_share + 2) / 2 = (younger_share + 2) / 2 := by
sorry

end sheep_problem_l1733_173359


namespace sum_of_reciprocals_is_six_l1733_173381

theorem sum_of_reciprocals_is_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : 
  1 / x + 1 / y = 6 := by
  sorry

end sum_of_reciprocals_is_six_l1733_173381


namespace max_value_of_inequality_l1733_173332

theorem max_value_of_inequality (x : ℝ) : 
  (∀ y : ℝ, (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) ≤ 0 → y ≤ x) → x = 1 := by
  sorry

end max_value_of_inequality_l1733_173332


namespace point_movement_on_number_line_l1733_173393

theorem point_movement_on_number_line :
  let start : ℤ := 0
  let move_right : ℤ := 2
  let move_left : ℤ := 8
  let final_position : ℤ := start + move_right - move_left
  final_position = -6 := by sorry

end point_movement_on_number_line_l1733_173393


namespace abc_inequality_l1733_173344

theorem abc_inequality (a b c : ℝ) (sum_zero : a + b + c = 0) (product_one : a * b * c = 1) :
  (a * b + b * c + c * a < 0) ∧ (max a (max b c) ≥ Real.rpow 4 (1/3)) := by
  sorry

end abc_inequality_l1733_173344


namespace solution_correctness_l1733_173378

theorem solution_correctness : ∀ x : ℝ,
  (((x^2 - 1)^2 - 5*(x^2 - 1) + 4 = 0) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 5 ∨ x = -Real.sqrt 5)) ∧
  ((x^4 - x^2 - 6 = 0) ↔ (x = Real.sqrt 3 ∨ x = -Real.sqrt 3)) := by
  sorry

end solution_correctness_l1733_173378


namespace triangle_angles_from_area_equation_l1733_173387

theorem triangle_angles_from_area_equation (α β γ : Real) (a b c : Real) (t : Real) :
  α = 43 * Real.pi / 180 →
  γ + β + α = Real.pi →
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β) →
  β = 17 * Real.pi / 180 ∧ γ = 120 * Real.pi / 180 := by
  sorry

end triangle_angles_from_area_equation_l1733_173387


namespace product_sum_max_l1733_173313

theorem product_sum_max (a b c d : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 5) (h4 : d = 6) :
  a * b + b * c + c * d + d * a = 63 := by sorry

end product_sum_max_l1733_173313


namespace complex_expression_equality_l1733_173355

theorem complex_expression_equality : ((7 - 3*I) - 3*(2 - 5*I)) * I = I - 12 := by
  sorry

end complex_expression_equality_l1733_173355


namespace unique_u_exists_l1733_173302

-- Define the variables as natural numbers
variable (a b u k p t : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = u
def condition2 : Prop := u + k = p
def condition3 : Prop := p + a = t
def condition4 : Prop := b + k + t = 20

-- Define the uniqueness condition
def unique_digits : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ u ≠ 0 ∧ k ≠ 0 ∧ p ≠ 0 ∧ t ≠ 0 ∧
  a ≠ b ∧ a ≠ u ∧ a ≠ k ∧ a ≠ p ∧ a ≠ t ∧
  b ≠ u ∧ b ≠ k ∧ b ≠ p ∧ b ≠ t ∧
  u ≠ k ∧ u ≠ p ∧ u ≠ t ∧
  k ≠ p ∧ k ≠ t ∧
  p ≠ t

-- Theorem statement
theorem unique_u_exists :
  ∃! u : ℕ, ∃ a b k p t : ℕ,
    condition1 a b u ∧
    condition2 u k p ∧
    condition3 p a t ∧
    condition4 b k t ∧
    unique_digits a b u k p t :=
  sorry

end unique_u_exists_l1733_173302


namespace part_one_part_two_l1733_173301

-- Define the equation
def equation (x a : ℝ) : Prop := (x + a) / (x - 2) - 5 / x = 1

-- Part 1: When x = 5 is a root
theorem part_one (a : ℝ) : (5 + a) / 3 - 1 = 1 → a = 1 := by sorry

-- Part 2: When the equation has no solution
theorem part_two (a : ℝ) : (∀ x : ℝ, ¬ equation x a) ↔ a = 3 ∨ a = -2 := by sorry

end part_one_part_two_l1733_173301


namespace smallest_of_five_consecutive_sum_100_l1733_173305

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    a + b + c + d + e = 100 ∧ 
    b = a + 1 ∧ 
    c = a + 2 ∧ 
    d = a + 3 ∧ 
    e = a + 4) → 
  n = 18 := by sorry

end smallest_of_five_consecutive_sum_100_l1733_173305


namespace initial_girls_count_l1733_173386

theorem initial_girls_count (b g : ℕ) : 
  (2 * (g - 15) = b) →
  (5 * (b - 45) = g - 15) →
  g = 40 := by
sorry

end initial_girls_count_l1733_173386


namespace chlorous_acid_weight_l1733_173303

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The atomic weight of Chlorine in g/mol -/
def Cl_weight : ℝ := 35.45

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Chlorous acid -/
def moles : ℝ := 6

/-- The molecular weight of Chlorous acid (HClO2) in g/mol -/
def HClO2_weight : ℝ := H_weight + Cl_weight + 2 * O_weight

/-- Theorem: The molecular weight of 6 moles of Chlorous acid (HClO2) is 410.76 grams -/
theorem chlorous_acid_weight : moles * HClO2_weight = 410.76 := by
  sorry

end chlorous_acid_weight_l1733_173303


namespace consecutive_composites_l1733_173333

theorem consecutive_composites
  (a t d r : ℕ+)
  (ha : ¬ Nat.Prime a.val)
  (ht : ¬ Nat.Prime t.val)
  (hd : ¬ Nat.Prime d.val)
  (hr : ¬ Nat.Prime r.val) :
  ∃ k : ℕ, ∀ i : ℕ, i < r → ¬ Nat.Prime (a * t ^ (k + i) + d) :=
sorry

end consecutive_composites_l1733_173333


namespace inequality_solution_l1733_173376

theorem inequality_solution (a x : ℝ) :
  (a * x^2 - (a + 3) * x + 3 ≤ 0) ↔
    (a < 0 ∧ (x ≤ 3/a ∨ x ≥ 1)) ∨
    (a = 0 ∧ x ≥ 1) ∨
    (0 < a ∧ a < 3 ∧ 1 ≤ x ∧ x ≤ 3/a) ∨
    (a = 3 ∧ x = 1) ∨
    (a > 3 ∧ 3/a ≤ x ∧ x ≤ 1) :=
by sorry

end inequality_solution_l1733_173376


namespace kira_breakfast_time_l1733_173392

/-- Represents the time taken to cook a single item -/
def cook_time (quantity : ℕ) (time_per_item : ℕ) : ℕ := quantity * time_per_item

/-- Represents Kira's breakfast preparation -/
def kira_breakfast : Prop :=
  let sausage_time := cook_time 3 5
  let egg_time := cook_time 6 4
  let bread_time := cook_time 4 3
  let hash_brown_time := cook_time 2 7
  let bacon_time := cook_time 4 6
  sausage_time + egg_time + bread_time + hash_brown_time + bacon_time = 89

theorem kira_breakfast_time : kira_breakfast := by
  sorry

end kira_breakfast_time_l1733_173392


namespace sum_of_roots_quadratic_l1733_173315

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end sum_of_roots_quadratic_l1733_173315


namespace problem_solution_l1733_173349

theorem problem_solution (x : ℝ) (h_pos : x > 0) :
  x^(2 * x^6) = 3 → x = (3 : ℝ)^(1/6) := by
  sorry

end problem_solution_l1733_173349


namespace existence_of_m_l1733_173316

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem existence_of_m :
  ∃ m : ℕ, 19 ≤ m ∧ m ≤ 60 ∧ 
  x m ≤ 4 + 1 / 2^10 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 4 + 1 / 2^10 :=
sorry

end existence_of_m_l1733_173316


namespace consecutive_product_square_appendage_l1733_173343

theorem consecutive_product_square_appendage (n : ℕ) :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ ∃ (k : ℕ), 100 * (n * (n + 1)) + 10 * a + b = k ^ 2 := by
  sorry

end consecutive_product_square_appendage_l1733_173343


namespace jed_speeding_fine_jed_speed_l1733_173369

theorem jed_speeding_fine (fine_per_mph : ℕ) (total_fine : ℕ) (speed_limit : ℕ) : ℕ :=
  let speed_over_limit := total_fine / fine_per_mph
  let total_speed := speed_limit + speed_over_limit
  total_speed

theorem jed_speed : jed_speeding_fine 16 256 50 = 66 := by
  sorry

end jed_speeding_fine_jed_speed_l1733_173369


namespace soccer_camp_ratio_l1733_173394

theorem soccer_camp_ratio :
  let total_kids : ℕ := 2000
  let soccer_kids : ℕ := total_kids / 2
  let afternoon_soccer_kids : ℕ := 750
  let morning_soccer_kids : ℕ := soccer_kids - afternoon_soccer_kids
  (morning_soccer_kids : ℚ) / (soccer_kids : ℚ) = 1 / 4 :=
by sorry

end soccer_camp_ratio_l1733_173394


namespace function_property_l1733_173314

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the constant k
variable (k : ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x : ℝ, f x + f (1 - x) = k)
                          (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
                          (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  k = 10 := by sorry

end function_property_l1733_173314


namespace triangle_side_expression_l1733_173356

theorem triangle_side_expression (a b c : ℝ) (h1 : a > c) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  |c - a| - Real.sqrt ((a + c - b) ^ 2) = b - 2 * c :=
sorry

end triangle_side_expression_l1733_173356


namespace equations_represent_scenario_l1733_173338

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℝ
  sheep : ℝ

/-- The system of equations representing the livestock values -/
def livestock_equations (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- The given scenario of livestock values -/
def livestock_scenario (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- Theorem stating that the system of equations correctly represents the scenario -/
theorem equations_represent_scenario :
  ∀ v : LivestockValue, livestock_equations v ↔ livestock_scenario v :=
by sorry

end equations_represent_scenario_l1733_173338


namespace prob_two_same_school_correct_l1733_173348

/-- Represents the number of schools participating in the activity -/
def num_schools : ℕ := 5

/-- Represents the number of students each school sends -/
def students_per_school : ℕ := 2

/-- Represents the total number of students participating -/
def total_students : ℕ := num_schools * students_per_school

/-- Represents the number of students chosen to play the game -/
def chosen_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the probability of exactly two students coming from the same school -/
def prob_two_same_school : ℚ := 5 / 14

theorem prob_two_same_school_correct :
  prob_two_same_school = 
    (choose num_schools 1 * choose students_per_school 2 * choose (total_students - students_per_school) 2) / 
    (choose total_students chosen_students) := by
  sorry

end prob_two_same_school_correct_l1733_173348


namespace unique_solution_quadratic_l1733_173371

theorem unique_solution_quadratic (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ n = 16 := by
sorry

end unique_solution_quadratic_l1733_173371


namespace line_not_parallel_intersects_plane_l1733_173340

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Definition: A line is parallel to a plane -/
def is_parallel (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Definition: A line shares common points with a plane -/
def shares_common_points (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is not parallel to a plane, then it shares common points with the plane -/
theorem line_not_parallel_intersects_plane (l : Line3D) (α : Plane3D) :
  ¬(is_parallel l α) → shares_common_points l α :=
by
  sorry

end line_not_parallel_intersects_plane_l1733_173340


namespace complex_modulus_product_l1733_173391

theorem complex_modulus_product : Complex.abs (4 - 3 * Complex.I) * Complex.abs (4 + 3 * Complex.I) = 25 := by
  sorry

end complex_modulus_product_l1733_173391


namespace max_value_of_cyclic_sum_l1733_173318

theorem max_value_of_cyclic_sum (a b c d e f : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f) 
  (sum_constraint : a + b + c + d + e + f = 6) : 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 := by
sorry

end max_value_of_cyclic_sum_l1733_173318


namespace set_operations_and_range_l1733_173379

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Theorem statement
theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 5}) ∧
  (B ∪ (Set.univ \ A) = {x | x ≤ 5 ∨ x ≥ 9}) ∧
  (∀ a : ℝ, C a ⊆ (Set.univ \ B) → (a < -4 ∨ a > 5)) :=
by sorry

end set_operations_and_range_l1733_173379


namespace rug_area_l1733_173351

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 8)
  (h_strip_width : strip_width = 2)
  (h_positive_length : floor_length > 0)
  (h_positive_width : floor_width > 0)
  (h_positive_strip : strip_width > 0)
  (h_strip_fits : 2 * strip_width < floor_length ∧ 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
sorry

end rug_area_l1733_173351


namespace diophantine_equation_solution_l1733_173309

theorem diophantine_equation_solution (x y z : ℕ+) :
  1 + 4^x.val + 4^y.val = z.val^2 ↔ 
  (∃ n : ℕ+, (x = n ∧ y = 2*n - 1 ∧ z = 1 + 2^(2*n.val - 1)) ∨ 
             (x = 2*n - 1 ∧ y = n ∧ z = 1 + 2^(2*n.val - 1))) :=
sorry

end diophantine_equation_solution_l1733_173309


namespace divisibility_of_fifth_power_differences_l1733_173300

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end divisibility_of_fifth_power_differences_l1733_173300


namespace highest_power_of_two_in_50_factorial_l1733_173399

theorem highest_power_of_two_in_50_factorial (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → (50 : ℕ).factorial % 2^k = 0) ∧ 
  (50 : ℕ).factorial % 2^(n + 1) ≠ 0 → 
  n = 47 := by
sorry

end highest_power_of_two_in_50_factorial_l1733_173399


namespace wire_cutting_l1733_173389

/-- Given a wire of length 80 cm, if it's cut into two pieces such that the longer piece
    is 3/5 of the shorter piece longer, then the length of the shorter piece is 400/13 cm. -/
theorem wire_cutting (total_length : ℝ) (shorter_piece : ℝ) :
  total_length = 80 ∧
  total_length = shorter_piece + (shorter_piece + 3/5 * shorter_piece) →
  shorter_piece = 400/13 := by
  sorry

end wire_cutting_l1733_173389


namespace flute_ratio_is_two_to_one_l1733_173374

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- The total number of instruments owned by a person -/
def total_instruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments in terms of F (number of flutes) -/
def carli (F : ℕ) : Instruments :=
  { flutes := F, horns := charlie.horns / 2, harps := 0 }

/-- The theorem to be proved -/
theorem flute_ratio_is_two_to_one :
  ∃ F : ℕ, 
    (total_instruments charlie + total_instruments (carli F) = 7) ∧ 
    ((carli F).flutes : ℚ) / charlie.flutes = 2 := by
  sorry

end flute_ratio_is_two_to_one_l1733_173374


namespace nikola_leaf_price_l1733_173328

/-- The price Nikola charges per leaf -/
def price_per_leaf : ℚ :=
  1 / 100

theorem nikola_leaf_price :
  let num_ants : ℕ := 400
  let food_per_ant : ℚ := 2
  let food_price : ℚ := 1 / 10
  let job_start_price : ℕ := 5
  let num_leaves : ℕ := 6000
  let num_jobs : ℕ := 4
  (↑num_jobs * job_start_price + ↑num_leaves * price_per_leaf : ℚ) =
    ↑num_ants * food_per_ant * food_price :=
by sorry

end nikola_leaf_price_l1733_173328


namespace ratio_equation_solution_l1733_173324

theorem ratio_equation_solution (x y z a : ℤ) : 
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  z = 30 * a - 15 →
  (∀ b : ℤ, 0 < b ∧ b < a → ¬(∃ (k : ℤ), 3 * k = 30 * b - 15)) →
  (∃ (k : ℤ), 3 * k = 30 * a - 15) →
  a = 4 := by
sorry

end ratio_equation_solution_l1733_173324


namespace hyperbola_center_is_correct_l1733_173334

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the given hyperbola is (3, 5) -/
theorem hyperbola_center_is_correct :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 5 - (y - hyperbola_center.2)^2 / (5/4) = 1) :=
by sorry

end hyperbola_center_is_correct_l1733_173334


namespace not_algebraic_expression_l1733_173320

-- Define what constitutes an algebraic expression
def is_algebraic_expression (e : Prop) : Prop :=
  ¬(∃ (x : ℝ), e ↔ x = 1)

-- Define the given expressions
def pi_expr : Prop := True
def x_equals_1 : Prop := ∃ (x : ℝ), x = 1
def one_over_x : Prop := True
def sqrt_3 : Prop := True

-- Theorem statement
theorem not_algebraic_expression :
  is_algebraic_expression pi_expr ∧
  is_algebraic_expression one_over_x ∧
  is_algebraic_expression sqrt_3 ∧
  ¬(is_algebraic_expression x_equals_1) :=
sorry

end not_algebraic_expression_l1733_173320


namespace cos_pi_minus_theta_point_l1733_173384

theorem cos_pi_minus_theta_point (θ : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos θ = 4 ∧ r * Real.sin θ = -3) →
  Real.cos (Real.pi - θ) = -4/5 := by
  sorry

end cos_pi_minus_theta_point_l1733_173384


namespace parallel_line_plane_false_l1733_173367

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Defines when a line is parallel to another line -/
def parallel_line_line (l1 l2 : Line) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_false :
  ∃ (a b : Line) (p : Plane),
    ¬(line_in_plane b p) ∧
    (line_in_plane a p) ∧
    (parallel_line_plane b p) ∧
    ¬(∀ (l : Line), line_in_plane l p → parallel_line_line b l) := by
  sorry

end parallel_line_plane_false_l1733_173367


namespace min_value_of_expression_l1733_173353

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ 3 ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end min_value_of_expression_l1733_173353


namespace gcd_of_256_180_600_l1733_173317

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end gcd_of_256_180_600_l1733_173317


namespace fish_swimming_north_l1733_173350

theorem fish_swimming_north (west east north caught_east caught_west left : ℕ) :
  west = 1800 →
  east = 3200 →
  caught_east = (2 * east) / 5 →
  caught_west = (3 * west) / 4 →
  left = 2870 →
  west + east + north = caught_east + caught_west + left →
  north = 500 := by
sorry

end fish_swimming_north_l1733_173350


namespace sequence_properties_l1733_173345

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ
  | 0 => 4
  | n + 1 => 2 * a n - 2 * (n + 1) + 1

/-- Definition of the sequence b_n -/
def b (t : ℝ) (n : ℕ) : ℝ := t * n + 2

/-- Theorem statement -/
theorem sequence_properties :
  (∀ n : ℕ, a (n + 1) - 2 * (n + 1) - 1 = 2 * (a n - 2 * n - 1)) ∧
  (∀ t : ℝ, (∀ n : ℕ, b t (n + 1) < 2 * a (n + 1)) → t < 6) := by
  sorry

end sequence_properties_l1733_173345


namespace quadratic_equations_solutions_l1733_173346

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 8*x - 6 = 0) ∧
  (∃ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0) ∧
  (∀ x : ℝ, x^2 - 8*x - 6 = 0 ↔ (x = 4 + Real.sqrt 22 ∨ x = 4 - Real.sqrt 22)) ∧
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ (x = 3 ∨ x = 1)) := by
  sorry

end quadratic_equations_solutions_l1733_173346


namespace four_six_eight_triangle_l1733_173327

/-- A predicate that determines if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 4, 6, and 8 can form a triangle -/
theorem four_six_eight_triangle :
  canFormTriangle 4 6 8 := by sorry

end four_six_eight_triangle_l1733_173327


namespace simplify_expression_l1733_173360

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

-- Additional definitions to match the problem conditions
def condition1 : (625 : ℝ) = 5^4 := by sorry
def condition2 : (343 : ℝ) = 7^3 := by sorry

end simplify_expression_l1733_173360


namespace reciprocal_of_negative_half_l1733_173398

theorem reciprocal_of_negative_half : ((-1/2 : ℚ)⁻¹ : ℚ) = -2 := by
  sorry

end reciprocal_of_negative_half_l1733_173398


namespace arithmetic_sequence_sum_l1733_173339

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) :=
by
  sorry


end arithmetic_sequence_sum_l1733_173339


namespace smallest_sum_of_a_and_b_l1733_173396

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 - a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 - 3*b*x + a = 0) : 
  a + b ≥ 3.3442 := by
sorry

end smallest_sum_of_a_and_b_l1733_173396


namespace polynomial_root_implies_coefficients_l1733_173310

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) :
  (∃ x : ℂ, x^3 + a*x^2 + 6*x + b = 0 ∧ x = 1 - 3*I) →
  a = 0 ∧ b = 20 := by
  sorry

end polynomial_root_implies_coefficients_l1733_173310


namespace susan_cloth_bags_l1733_173365

/-- Calculates the number of cloth bags Susan brought to carry peaches. -/
def number_of_cloth_bags (total_peaches knapsack_peaches : ℕ) : ℕ :=
  let cloth_bag_peaches := 2 * knapsack_peaches
  (total_peaches - knapsack_peaches) / cloth_bag_peaches

/-- Proves that Susan brought 2 cloth bags given the problem conditions. -/
theorem susan_cloth_bags :
  number_of_cloth_bags (5 * 12) 12 = 2 := by
  sorry

end susan_cloth_bags_l1733_173365


namespace correct_calculation_l1733_173323

theorem correct_calculation : ∃ x : ℕ, (x + 30 = 86) ∧ (x * 30 = 1680) := by
  sorry

end correct_calculation_l1733_173323


namespace min_value_theorem_l1733_173385

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y, x > 0 → y > 0 → x + 1/y = 2 → 2/x + 2*y ≥ m := by
  sorry

end min_value_theorem_l1733_173385


namespace walking_speed_problem_l1733_173325

/-- Two people A and B walk towards each other and meet. This theorem proves B's speed. -/
theorem walking_speed_problem (speed_A speed_B : ℝ) (initial_distance total_time : ℝ) : 
  speed_A = 5 →
  initial_distance = 24 →
  total_time = 2 →
  speed_A * total_time + speed_B * total_time = initial_distance →
  speed_B = 7 := by
  sorry

#check walking_speed_problem

end walking_speed_problem_l1733_173325


namespace g_4_cubed_eq_16_l1733_173364

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^3 = 16 -/
theorem g_4_cubed_eq_16
  (f g : ℝ → ℝ)
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^3)
  (h3 : g 16 = 16) :
  (g 4)^3 = 16 := by
  sorry

end g_4_cubed_eq_16_l1733_173364


namespace matrix_power_2023_l1733_173373

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l1733_173373


namespace jacqueline_apples_l1733_173322

/-- The number of plums Jacqueline had initially -/
def plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def guavas : ℕ := 18

/-- The number of fruits Jacqueline gave to Jane -/
def given_fruits : ℕ := 40

/-- The number of fruits Jacqueline had left after giving some to Jane -/
def left_fruits : ℕ := 15

/-- The number of apples Jacqueline had initially -/
def apples : ℕ := 21

theorem jacqueline_apples :
  plums + guavas + apples = given_fruits + left_fruits :=
sorry

end jacqueline_apples_l1733_173322


namespace line_passes_through_point_two_two_l1733_173306

/-- The line equation is of the form (1+4k)x-(2-3k)y+2-14k=0 where k is a real parameter -/
def line_equation (k x y : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + 2 - 14*k = 0

/-- Theorem: The line passes through the point (2, 2) for all values of k -/
theorem line_passes_through_point_two_two :
  ∀ k : ℝ, line_equation k 2 2 := by sorry

end line_passes_through_point_two_two_l1733_173306


namespace ice_cream_bill_l1733_173375

theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) : 
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 → 
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 := by
  sorry

#check ice_cream_bill

end ice_cream_bill_l1733_173375


namespace product_closure_infinite_pairs_l1733_173377

/-- The set M of integers of the form a^2 + 13b^2, where a and b are nonzero integers -/
def M : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ n = a^2 + 13*b^2}

/-- The product of any two elements of M is an element of M -/
theorem product_closure (m1 m2 : ℤ) (h1 : m1 ∈ M) (h2 : m2 ∈ M) : m1 * m2 ∈ M := by
  sorry

/-- Definition of the sequence xk -/
def x (k : ℕ) : ℤ := (2^13 + 1) * ((4*k)^2 + 13*(4*k + 1)^2)

/-- Definition of the sequence yk -/
def y (k : ℕ) : ℤ := 2 * x k

/-- There are infinitely many pairs (x, y) such that x + y ∉ M but x^13 + y^13 ∈ M -/
theorem infinite_pairs : ∀ k : ℕ, (x k + y k ∉ M) ∧ ((x k)^13 + (y k)^13 ∈ M) := by
  sorry

end product_closure_infinite_pairs_l1733_173377


namespace trail_mix_weight_l1733_173307

/-- The weight of peanuts in pounds -/
def weight_peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def weight_chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def weight_raisins : ℝ := 0.08

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := weight_peanuts + weight_chocolate_chips + weight_raisins

theorem trail_mix_weight : total_weight = 0.42 := by
  sorry

end trail_mix_weight_l1733_173307


namespace students_without_a_count_l1733_173329

/-- Represents the number of students in a school course with various grade distributions. -/
structure CourseData where
  total_students : ℕ
  history_as : ℕ
  math_as : ℕ
  both_as : ℕ
  math_only_a : ℕ
  history_only_attendees : ℕ

/-- Calculates the number of students who did not receive an A in either class. -/
def students_without_a (data : CourseData) : ℕ :=
  data.total_students - (data.history_as + data.math_as - data.both_as)

/-- Theorem stating the number of students who did not receive an A in either class. -/
theorem students_without_a_count (data : CourseData) 
  (h1 : data.total_students = 30)
  (h2 : data.history_only_attendees = 1)
  (h3 : data.history_as = 6)
  (h4 : data.math_as = 15)
  (h5 : data.both_as = 3)
  (h6 : data.math_only_a = 1) :
  students_without_a data = 12 := by
  sorry

#eval students_without_a {
  total_students := 30,
  history_as := 6,
  math_as := 15,
  both_as := 3,
  math_only_a := 1,
  history_only_attendees := 1
}

end students_without_a_count_l1733_173329


namespace tangent_line_determines_coefficients_l1733_173388

theorem tangent_line_determines_coefficients :
  ∀ (a b : ℝ),
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let tangent_line : ℝ → ℝ := λ x => x + 1
  (f 0 = 1) →
  (∀ x, tangent_line x = x - f x + 1) →
  (∀ h : ℝ, h ≠ 0 → (f h - f 0) / h = tangent_line 0) →
  a = 1 ∧ b = 1 := by
sorry

end tangent_line_determines_coefficients_l1733_173388


namespace smallest_b_in_geometric_sequence_l1733_173330

/-- 
Given a geometric sequence of positive terms a, b, c with product 216,
this theorem states that the smallest possible value of b is 6.
-/
theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- all terms are positive
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r) →  -- geometric sequence
  a * b * c = 216 →  -- product is 216
  (∀ b' : ℝ, b' > 0 → 
    (∃ a' c' : ℝ, a' > 0 ∧ c' > 0 ∧ 
      (∃ r : ℝ, r > 0 ∧ b' = a' * r ∧ c' = b' * r) ∧ 
      a' * b' * c' = 216) → 
    b' ≥ 6) →  -- for any valid b', b' is at least 6
  b = 6  -- therefore, the smallest possible b is 6
:= by sorry

end smallest_b_in_geometric_sequence_l1733_173330


namespace pencil_count_in_10x10_grid_l1733_173390

/-- Represents a grid of items -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of items on the perimeter of a grid -/
def perimeterCount (g : Grid) : ℕ :=
  2 * (g.rows + g.cols) - 4

/-- Calculates the number of items inside a grid (excluding the perimeter) -/
def innerCount (g : Grid) : ℕ :=
  (g.rows - 2) * (g.cols - 2)

/-- The main theorem stating that in a 10x10 grid, the number of pencils inside is 64 -/
theorem pencil_count_in_10x10_grid :
  let g : Grid := { rows := 10, cols := 10 }
  innerCount g = 64 := by sorry

end pencil_count_in_10x10_grid_l1733_173390


namespace first_term_of_geometric_sequence_l1733_173368

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem first_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end first_term_of_geometric_sequence_l1733_173368


namespace triangle_area_l1733_173331

theorem triangle_area (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (base * height) / 2 = 16 := by
  sorry

end triangle_area_l1733_173331


namespace clearance_sale_gain_percentage_l1733_173335

-- Define the original selling price
def original_selling_price : ℝ := 30

-- Define the original gain percentage
def original_gain_percentage : ℝ := 20

-- Define the discount percentage during clearance sale
def clearance_discount_percentage : ℝ := 10

-- Theorem statement
theorem clearance_sale_gain_percentage :
  let cost_price := original_selling_price / (1 + original_gain_percentage / 100)
  let discounted_price := original_selling_price * (1 - clearance_discount_percentage / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percentage := (new_gain / cost_price) * 100
  new_gain_percentage = 8 := by sorry

end clearance_sale_gain_percentage_l1733_173335


namespace solve_allowance_problem_l1733_173342

def allowance_problem (initial_amount spent_amount final_amount : ℕ) : Prop :=
  ∃ allowance : ℕ, 
    initial_amount - spent_amount + allowance = final_amount

theorem solve_allowance_problem :
  allowance_problem 5 2 8 → ∃ allowance : ℕ, allowance = 5 :=
by
  sorry

end solve_allowance_problem_l1733_173342


namespace hyperbola_equation_l1733_173382

/-- A hyperbola with center at the origin, focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
  let midpoint : ℝ × ℝ := (0, 2)
  (P.1^2 / 1^2 - P.2^2 / 4^2 = 1) ∧ 
  ((P.1 + F₁.1) / 2 = midpoint.1 ∧ (P.2 + F₁.2) / 2 = midpoint.2) →
  ∀ x y : ℝ, x^2 - y^2/4 = 1 ↔ (x^2 / 1^2 - y^2 / 4^2 = 1) := by
  sorry

#check hyperbola_equation

end hyperbola_equation_l1733_173382


namespace range_of_a_l1733_173304

def S (a : ℝ) := {x : ℝ | x^2 ≤ a}

theorem range_of_a (a : ℝ) : (∅ ⊂ S a) → a ∈ Set.Ici 0 := by
  sorry

end range_of_a_l1733_173304


namespace square_root_of_3_plus_4i_l1733_173383

theorem square_root_of_3_plus_4i :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I ∧
  (-2 - Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I :=
by
  sorry

end square_root_of_3_plus_4i_l1733_173383


namespace part_one_evaluation_part_two_evaluation_l1733_173321

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part I
theorem part_one_evaluation : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Part II
theorem part_two_evaluation :
  lg 14 - 2 * lg (7/3) + lg 7 - lg 18 = 0 := by sorry

end part_one_evaluation_part_two_evaluation_l1733_173321


namespace simplify_sqrt_one_minus_sin_twenty_deg_l1733_173308

theorem simplify_sqrt_one_minus_sin_twenty_deg :
  Real.sqrt (1 - Real.sin (20 * π / 180)) = Real.cos (10 * π / 180) - Real.sin (10 * π / 180) := by
  sorry

end simplify_sqrt_one_minus_sin_twenty_deg_l1733_173308


namespace least_meeting_time_for_four_horses_l1733_173326

def horse_lap_time (k : Nat) : Nat := 2 * k

def is_meeting_time (t : Nat) (horses : List Nat) : Prop :=
  ∀ h ∈ horses, t % (horse_lap_time h) = 0

theorem least_meeting_time_for_four_horses :
  ∃ T : Nat,
    T > 0 ∧
    (∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time T horses) ∧
    (∀ t : Nat, 0 < t ∧ t < T →
      ¬∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time t horses) ∧
    T = 24 := by sorry

end least_meeting_time_for_four_horses_l1733_173326


namespace max_sum_on_circle_l1733_173370

theorem max_sum_on_circle : 
  ∀ x y : ℤ, 
  x^2 + y^2 = 169 → 
  x ≥ y → 
  x + y ≤ 21 := by
sorry

end max_sum_on_circle_l1733_173370


namespace pants_price_l1733_173358

/-- The selling price of a pair of pants given the price of a coat and the discount percentage -/
theorem pants_price (coat_price : ℝ) (discount_percent : ℝ) (pants_price : ℝ) : 
  coat_price = 800 →
  discount_percent = 40 →
  pants_price = coat_price * (1 - discount_percent / 100) →
  pants_price = 480 := by
sorry

end pants_price_l1733_173358


namespace angle_sum_is_pi_over_two_l1733_173361

theorem angle_sum_is_pi_over_two 
  (α β γ : Real) 
  (h_sin_α : Real.sin α = 1/3)
  (h_sin_β : Real.sin β = 1/(3*Real.sqrt 11))
  (h_sin_γ : Real.sin γ = 3/Real.sqrt 11)
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_acute_γ : 0 < γ ∧ γ < π/2) :
  α + β + γ = π/2 := by
sorry

end angle_sum_is_pi_over_two_l1733_173361


namespace percentage_saved_approximately_11_percent_l1733_173395

def original_price : ℝ := 49.50
def spent_amount : ℝ := 44.00
def saved_amount : ℝ := original_price - spent_amount

theorem percentage_saved_approximately_11_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  (saved_amount / original_price) * 100 ∈ Set.Icc (11 - ε) (11 + ε) := by
sorry

end percentage_saved_approximately_11_percent_l1733_173395


namespace square_pyramid_sum_l1733_173363

/-- A square pyramid is a polyhedron with a square base and four triangular faces. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat

/-- The sum of faces, edges, and vertices for a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

end square_pyramid_sum_l1733_173363


namespace equation_solutions_l1733_173336

theorem equation_solutions :
  (∃ x : ℝ, 2 * x^3 = 16 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1) :=
by sorry

end equation_solutions_l1733_173336


namespace set_equality_l1733_173362

theorem set_equality (M : Set ℕ) : M ∪ {1} = {1, 2, 3} → M = {2, 3} := by
  sorry

end set_equality_l1733_173362


namespace intersection_of_A_and_B_l1733_173397

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 9}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l1733_173397


namespace new_hires_all_women_l1733_173380

theorem new_hires_all_women 
  (initial_workers : ℕ) 
  (new_hires : ℕ) 
  (initial_men_fraction : ℚ) 
  (final_women_percentage : ℚ) :
  initial_workers = 90 →
  new_hires = 10 →
  initial_men_fraction = 2/3 →
  final_women_percentage = 40/100 →
  (initial_workers * (1 - initial_men_fraction) + new_hires) / (initial_workers + new_hires) = final_women_percentage →
  new_hires / new_hires = 1 :=
by sorry

end new_hires_all_women_l1733_173380


namespace factor_theorem_application_l1733_173352

theorem factor_theorem_application (m k : ℝ) : 
  (∃ q : ℝ, m^2 - k*m - 24 = (m - 8) * q) → k = 5 := by
  sorry

end factor_theorem_application_l1733_173352


namespace double_in_fifty_years_l1733_173312

/-- The interest rate (in percentage) that doubles an initial sum in 50 years under simple interest -/
def double_interest_rate : ℝ := 2

theorem double_in_fifty_years (P : ℝ) (P_pos : P > 0) :
  P * (1 + double_interest_rate * 50 / 100) = 2 * P := by
  sorry

#check double_in_fifty_years

end double_in_fifty_years_l1733_173312


namespace apple_price_per_kg_final_apple_price_is_correct_l1733_173341

/-- Calculates the final price per kilogram of apples after discounts -/
theorem apple_price_per_kg (weight : ℝ) (original_price : ℝ) 
  (discount_percent : ℝ) (volume_discount_percent : ℝ) 
  (volume_discount_threshold : ℝ) : ℝ :=
  let price_after_discount := original_price * (1 - discount_percent)
  let final_price := 
    if weight > volume_discount_threshold
    then price_after_discount * (1 - volume_discount_percent)
    else price_after_discount
  final_price / weight

/-- Proves that the final price per kilogram is $1.44 given the specific conditions -/
theorem final_apple_price_is_correct : 
  apple_price_per_kg 5 10 0.2 0.1 3 = 1.44 := by
  sorry

end apple_price_per_kg_final_apple_price_is_correct_l1733_173341


namespace peter_age_is_16_l1733_173311

/-- Peter's present age -/
def PeterAge : ℕ := sorry

/-- Jacob's present age -/
def JacobAge : ℕ := sorry

/-- Theorem stating the conditions and the result to prove -/
theorem peter_age_is_16 :
  (JacobAge = PeterAge + 12) ∧
  (PeterAge - 10 = (JacobAge - 10) / 3) →
  PeterAge = 16 := by sorry

end peter_age_is_16_l1733_173311
