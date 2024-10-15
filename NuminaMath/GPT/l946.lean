import Mathlib

namespace NUMINAMATH_GPT_wage_percent_change_l946_94609

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end NUMINAMATH_GPT_wage_percent_change_l946_94609


namespace NUMINAMATH_GPT_incorrect_statement_A_l946_94604

-- We need to prove that statement (A) is incorrect given the provided conditions.

theorem incorrect_statement_A :
  ¬(∀ (a b : ℝ), a > b → ∀ (c : ℝ), c < 0 → a * c > b * c ∧ a / c > b / c) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_A_l946_94604


namespace NUMINAMATH_GPT_select_two_people_l946_94661

theorem select_two_people {n : ℕ} (h1 : n ≠ 0) (h2 : n ≥ 2) (h3 : (n - 1) ^ 2 = 25) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_select_two_people_l946_94661


namespace NUMINAMATH_GPT_find_angle_A_l946_94620

theorem find_angle_A (a b : ℝ) (sin_B : ℝ) (ha : a = 3) (hb : b = 4) (hsinB : sin_B = 2/3) :
  ∃ A : ℝ, A = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l946_94620


namespace NUMINAMATH_GPT_second_set_parallel_lines_l946_94621

theorem second_set_parallel_lines (n : ℕ) :
  (5 * (n - 1)) = 280 → n = 71 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_second_set_parallel_lines_l946_94621


namespace NUMINAMATH_GPT_distance_traveled_by_second_hand_l946_94699

theorem distance_traveled_by_second_hand (r : ℝ) (minutes : ℝ) (h1 : r = 10) (h2 : minutes = 45) :
  (2 * Real.pi * r) * (minutes / 1) = 900 * Real.pi := by
  -- Given:
  -- r = length of the second hand = 10 cm
  -- minutes = 45
  -- To prove: distance traveled by the tip = 900π cm
  sorry

end NUMINAMATH_GPT_distance_traveled_by_second_hand_l946_94699


namespace NUMINAMATH_GPT_solve_for_x_l946_94652

noncomputable def simplified_end_expr (x : ℝ) := x = 4 - Real.sqrt 7 
noncomputable def expressed_as_2_statement (x : ℝ) := (x ^ 2 - 4 * x + 5) = (4 * (x - 1))
noncomputable def domain_condition (x : ℝ) := (-5 < x) ∧ (x < 3)

theorem solve_for_x (x : ℝ) :
  domain_condition x →
  (expressed_as_2_statement x ↔ simplified_end_expr x) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l946_94652


namespace NUMINAMATH_GPT_teacherZhangAge_in_5_years_correct_l946_94662

variable (a : ℕ)

def teacherZhangAgeCurrent := 3 * a - 2

def teacherZhangAgeIn5Years := teacherZhangAgeCurrent a + 5

theorem teacherZhangAge_in_5_years_correct :
  teacherZhangAgeIn5Years a = 3 * a + 3 := by
  sorry

end NUMINAMATH_GPT_teacherZhangAge_in_5_years_correct_l946_94662


namespace NUMINAMATH_GPT_rectangle_area_l946_94654

variable (a b c : ℝ)

theorem rectangle_area (h : a^2 + b^2 = c^2) : a * b = area :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l946_94654


namespace NUMINAMATH_GPT_jason_total_spent_l946_94637

theorem jason_total_spent (h_shorts : ℝ) (h_jacket : ℝ) (h1 : h_shorts = 14.28) (h2 : h_jacket = 4.74) : h_shorts + h_jacket = 19.02 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_jason_total_spent_l946_94637


namespace NUMINAMATH_GPT_child_ticket_cost_l946_94689

theorem child_ticket_cost :
  ∃ x : ℤ, (9 * 11 = 7 * x + 50) ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l946_94689


namespace NUMINAMATH_GPT_find_x_l946_94683

theorem find_x (x : ℝ) (h : 0 < x) (hx : 0.01 * x * x^2 = 16) : x = 12 :=
sorry

end NUMINAMATH_GPT_find_x_l946_94683


namespace NUMINAMATH_GPT_multiply_exponents_l946_94666

variable (a : ℝ)

theorem multiply_exponents :
  a * a^2 * (-a)^3 = -a^6 := 
sorry

end NUMINAMATH_GPT_multiply_exponents_l946_94666


namespace NUMINAMATH_GPT_train_speed_l946_94634

/-- Proof problem: Speed calculation of a train -/
theorem train_speed :
  ∀ (length : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ),
    length = 40 →
    time_seconds = 0.9999200063994881 →
    speed_kmph = (length / 1000) / (time_seconds / 3600) →
    speed_kmph = 144 :=
by
  intros length time_seconds speed_kmph h_length h_time_seconds h_speed_kmph
  rw [h_length, h_time_seconds] at h_speed_kmph
  -- sorry is used to skip the proof steps
  sorry 

end NUMINAMATH_GPT_train_speed_l946_94634


namespace NUMINAMATH_GPT_total_triangles_in_geometric_figure_l946_94691

noncomputable def numberOfTriangles : ℕ :=
  let smallest_triangles := 3 + 2 + 1
  let medium_triangles := 2
  let large_triangle := 1
  smallest_triangles + medium_triangles + large_triangle

theorem total_triangles_in_geometric_figure : numberOfTriangles = 9 := by
  unfold numberOfTriangles
  sorry

end NUMINAMATH_GPT_total_triangles_in_geometric_figure_l946_94691


namespace NUMINAMATH_GPT_price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l946_94606

def cost_price : ℝ := 40
def initial_price : ℝ := 60
def initial_sales_volume : ℕ := 300
def sales_decrease_rate (x : ℕ) : ℕ := 10 * x
def sales_increase_rate (a : ℕ) : ℕ := 20 * a

noncomputable def price_increase_proft_relation (x : ℕ) : ℝ :=
  -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000

theorem price_increase_profit_relation_proof (x : ℕ) (h : 0 ≤ x ∧ x ≤ 30) :
  price_increase_proft_relation x = -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000 := sorry

noncomputable def price_decrease_profit_relation (a : ℕ) : ℝ :=
  -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000

theorem price_decrease_profit_relation_proof (a : ℕ) :
  price_decrease_profit_relation a = -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000 := sorry

theorem max_profit_price_increase :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧ price_increase_proft_relation x = 6250 := sorry

end NUMINAMATH_GPT_price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l946_94606


namespace NUMINAMATH_GPT_domain_of_f_l946_94617

noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -5} := 
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l946_94617


namespace NUMINAMATH_GPT_min_value_am_gm_l946_94673

theorem min_value_am_gm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_am_gm_l946_94673


namespace NUMINAMATH_GPT_number_difference_l946_94623

theorem number_difference 
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 2 * a2)
  (h2 : a1 = 3 * a3)
  (h3 : (a1 + a2 + a3) / 3 = 88) : 
  a1 - a3 = 96 :=
sorry

end NUMINAMATH_GPT_number_difference_l946_94623


namespace NUMINAMATH_GPT_part1_part2_l946_94658

-- Definitions and conditions
def prop_p (a : ℝ) : Prop := 
  let Δ := -4 * a^2 + 4 * a + 24 
  Δ ≥ 0

def neg_prop_p (a : ℝ) : Prop := ¬ prop_p a

def prop_q (m a : ℝ) : Prop := 
  (m - 1 ≤ a ∧ a ≤ m + 3)

-- Part 1 theorem statement
theorem part1 (a : ℝ) : neg_prop_p a → (a < -2 ∨ a > 3) :=
by sorry

-- Part 2 theorem statement
theorem part2 (m : ℝ) : 
  (∀ a : ℝ, prop_q m a → prop_p a) ∧ (∃ a : ℝ, prop_p a ∧ ¬ prop_q m a) → (-1 ≤ m ∧ m < 0) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l946_94658


namespace NUMINAMATH_GPT_distance_is_correct_l946_94653

noncomputable def distance_from_center_to_plane
  (O : Point)
  (radius : ℝ)
  (vertices : Point × Point × Point)
  (side_lengths : (ℝ × ℝ × ℝ)) :
  ℝ :=
  8.772

theorem distance_is_correct
  (O : Point)
  (radius : ℝ)
  (A B C : Point)
  (h_radius : radius = 10)
  (h_sides : side_lengths = (17, 17, 16))
  (vertices := (A, B, C)) :
  distance_from_center_to_plane O radius vertices side_lengths = 8.772 := by
  sorry

end NUMINAMATH_GPT_distance_is_correct_l946_94653


namespace NUMINAMATH_GPT_range_of_a_l946_94696

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1) < a ∧ a ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l946_94696


namespace NUMINAMATH_GPT_ticket_cost_is_25_l946_94600

-- Define the given conditions
def num_tickets_first_show : ℕ := 200
def num_tickets_second_show : ℕ := 3 * num_tickets_first_show
def total_tickets : ℕ := num_tickets_first_show + num_tickets_second_show
def total_revenue_in_dollars : ℕ := 20000

-- Claim to prove
theorem ticket_cost_is_25 : ∃ x : ℕ, total_tickets * x = total_revenue_in_dollars ∧ x = 25 :=
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_ticket_cost_is_25_l946_94600


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_5_l946_94685

theorem remainder_when_sum_divided_by_5 (f y : ℤ) (k m : ℤ) 
  (hf : f = 5 * k + 3) (hy : y = 5 * m + 4) : 
  (f + y) % 5 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_sum_divided_by_5_l946_94685


namespace NUMINAMATH_GPT_cargo_loaded_in_bahamas_l946_94679

def initial : ℕ := 5973
def final : ℕ := 14696
def loaded : ℕ := final - initial

theorem cargo_loaded_in_bahamas : loaded = 8723 := by
  sorry

end NUMINAMATH_GPT_cargo_loaded_in_bahamas_l946_94679


namespace NUMINAMATH_GPT_chloe_candies_l946_94601

-- Definitions for the conditions
def lindaCandies : ℕ := 34
def totalCandies : ℕ := 62

-- The statement to prove
theorem chloe_candies :
  (totalCandies - lindaCandies) = 28 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_chloe_candies_l946_94601


namespace NUMINAMATH_GPT_problem_I_problem_II_l946_94636

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Problem (I): Prove solution set
theorem problem_I (x : ℝ) : f (x - 1) + f (x + 3) ≥ 6 ↔ (x ≤ -3 ∨ x ≥ 3) := by
  sorry

-- Problem (II): Prove inequality given conditions
theorem problem_II (a b : ℝ) (ha: |a| < 1) (hb: |b| < 1) (hano: a ≠ 0) : 
  f (a * b) > |a| * f (b / a) := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l946_94636


namespace NUMINAMATH_GPT_quadrilateral_has_four_sides_and_angles_l946_94682

-- Define the conditions based on the characteristics of a quadrilateral
def quadrilateral (sides angles : Nat) : Prop :=
  sides = 4 ∧ angles = 4

-- Statement: Verify the property of a quadrilateral
theorem quadrilateral_has_four_sides_and_angles (sides angles : Nat) (h : quadrilateral sides angles) : sides = 4 ∧ angles = 4 :=
by
  -- We provide a proof by the characteristics of a quadrilateral
  sorry

end NUMINAMATH_GPT_quadrilateral_has_four_sides_and_angles_l946_94682


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l946_94619

noncomputable def side_length_of_triangle (PQ PR PS : ℕ) : ℝ := 
  let s := 8 * Real.sqrt 3
  s

theorem equilateral_triangle_side_length (PQ PR PS : ℕ) (P_inside_triangle : true) 
  (Q_foot : true) (R_foot : true) (S_foot : true)
  (hPQ : PQ = 2) (hPR : PR = 4) (hPS : PS = 6) : 
  side_length_of_triangle PQ PR PS = 8 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l946_94619


namespace NUMINAMATH_GPT_rectangle_perimeter_is_22_l946_94615

-- Definition of sides of the triangle DEF
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Helper function to compute the area of a right triangle
def triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Ensure the triangle is a right triangle and calculate its area
def area_of_triangle : ℕ :=
  if (side1 * side1 + side2 * side2 = hypotenuse * hypotenuse) then
    triangle_area side1 side2
  else
    0

-- Definition of rectangle's width and equation to find its perimeter
def width : ℕ := 5
def rectangle_length : ℕ := area_of_triangle / width
def perimeter_of_rectangle : ℕ := 2 * (width + rectangle_length)

theorem rectangle_perimeter_is_22 : perimeter_of_rectangle = 22 :=
by
  -- Proof content goes here
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_22_l946_94615


namespace NUMINAMATH_GPT_each_friend_gave_bella_2_roses_l946_94665

-- Define the given conditions
def total_roses_from_parents : ℕ := 2 * 12
def total_roses_bella_received : ℕ := 44
def number_of_dancer_friends : ℕ := 10

-- Define the mathematical goal
def roses_from_each_friend (total_roses_from_parents total_roses_bella_received number_of_dancer_friends : ℕ) : ℕ :=
  (total_roses_bella_received - total_roses_from_parents) / number_of_dancer_friends

-- Prove that each dancer friend gave Bella 2 roses
theorem each_friend_gave_bella_2_roses :
  roses_from_each_friend total_roses_from_parents total_roses_bella_received number_of_dancer_friends = 2 :=
by
  sorry

end NUMINAMATH_GPT_each_friend_gave_bella_2_roses_l946_94665


namespace NUMINAMATH_GPT_dried_mushrooms_weight_l946_94684

theorem dried_mushrooms_weight (fresh_weight : ℝ) (water_content_fresh : ℝ) (water_content_dried : ℝ) :
  fresh_weight = 22 →
  water_content_fresh = 0.90 →
  water_content_dried = 0.12 →
  ∃ x : ℝ, x = 2.5 :=
by
  intros h1 h2 h3
  have hw_fresh : ℝ := fresh_weight * water_content_fresh
  have dry_material_fresh : ℝ := fresh_weight - hw_fresh
  have dry_material_dried : ℝ := 1.0 - water_content_dried
  have hw_dried := dry_material_fresh / dry_material_dried
  use hw_dried
  sorry

end NUMINAMATH_GPT_dried_mushrooms_weight_l946_94684


namespace NUMINAMATH_GPT_no_valid_a_exists_l946_94605

theorem no_valid_a_exists 
  (a : ℝ)
  (h1: ∀ x : ℝ, x^2 + 2*(a+1)*x - (a-1) = 0 → (1 < x ∨ x < 1)) :
  false := by
  sorry

end NUMINAMATH_GPT_no_valid_a_exists_l946_94605


namespace NUMINAMATH_GPT_min_value_of_expression_l946_94635

noncomputable def given_expression (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2)

theorem min_value_of_expression : ∃ x : ℝ, given_expression x = 6 * Real.sqrt 2 := 
by 
  use 0
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l946_94635


namespace NUMINAMATH_GPT_element_in_set_l946_94610

theorem element_in_set (A : Set ℕ) (h : A = {1, 2}) : 1 ∈ A := 
by 
  rw[h]
  simp

end NUMINAMATH_GPT_element_in_set_l946_94610


namespace NUMINAMATH_GPT_problem_statement_false_adjacent_complementary_l946_94611

-- Definition of straight angle, supplementary angles, and complementary angles.
def is_straight_angle (θ : ℝ) : Prop := θ = 180
def are_supplementary (θ ψ : ℝ) : Prop := θ + ψ = 180
def are_complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Definition of adjacent angles (for completeness, though we don't use adjacency differently right now)
def are_adjacent (θ ψ : ℝ) : Prop := ∀ x, θ + x + ψ + x = θ + ψ -- Simplified

-- Additional conditions that could be true or false -- we need one of them to be false.
def false_statement_D (θ ψ : ℝ) : Prop :=
  are_complementary θ ψ → are_adjacent θ ψ

theorem problem_statement_false_adjacent_complementary :
  ∃ (θ ψ : ℝ), ¬ false_statement_D θ ψ :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_false_adjacent_complementary_l946_94611


namespace NUMINAMATH_GPT_probability_same_length_l946_94622

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end NUMINAMATH_GPT_probability_same_length_l946_94622


namespace NUMINAMATH_GPT_total_juice_boxes_needed_l946_94656

-- Definitions for the conditions
def john_juice_per_week : Nat := 2 * 5
def john_school_weeks : Nat := 18 - 2 -- taking into account the holiday break

def samantha_juice_per_week : Nat := 1 * 5
def samantha_school_weeks : Nat := 16 - 2 -- taking into account after-school and holiday break

def heather_mon_wed_juice : Nat := 3 * 2
def heather_tue_thu_juice : Nat := 2 * 2
def heather_fri_juice : Nat := 1
def heather_juice_per_week : Nat := heather_mon_wed_juice + heather_tue_thu_juice + heather_fri_juice
def heather_school_weeks : Nat := 17 - 2 -- taking into account personal break and holiday break

-- Question and Answer in lean
theorem total_juice_boxes_needed : 
  (john_juice_per_week * john_school_weeks) + 
  (samantha_juice_per_week * samantha_school_weeks) + 
  (heather_juice_per_week * heather_school_weeks) = 395 := 
by
  sorry

end NUMINAMATH_GPT_total_juice_boxes_needed_l946_94656


namespace NUMINAMATH_GPT_winner_won_by_l946_94663

theorem winner_won_by (V : ℝ) (h₁ : 0.62 * V = 806) : 806 - 0.38 * V = 312 :=
by
  sorry

end NUMINAMATH_GPT_winner_won_by_l946_94663


namespace NUMINAMATH_GPT_functional_equation_solution_l946_94698

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l946_94698


namespace NUMINAMATH_GPT_area_relation_l946_94695

open Real

noncomputable def S_OMN (a b c d θ : ℝ) : ℝ := 1 / 2 * abs (b * c - a * d) * sin θ
noncomputable def S_ABCD (a b c d θ : ℝ) : ℝ := 2 * abs (b * c - a * d) * sin θ

theorem area_relation (a b c d θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
    4 * (S_OMN a b c d θ) = S_ABCD a b c d θ :=
by
  sorry

end NUMINAMATH_GPT_area_relation_l946_94695


namespace NUMINAMATH_GPT_range_of_a_l946_94608

noncomputable def exists_unique_y (a : ℝ) (x : ℝ) : Prop :=
∃! (y : ℝ), y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y = a

theorem range_of_a (e : ℝ) (H_e : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 0 1, exists_unique_y a x) →
  a ∈ Set.Ioc (1 + 1/e) e :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l946_94608


namespace NUMINAMATH_GPT_find_omega2019_value_l946_94648

noncomputable def omega_n (n : ℕ) : ℝ := (2 * n - 1) * Real.pi / 2

theorem find_omega2019_value :
  omega_n 2019 = 4037 * Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_omega2019_value_l946_94648


namespace NUMINAMATH_GPT_diana_erasers_l946_94639

theorem diana_erasers (number_of_friends : ℕ) (erasers_per_friend : ℕ) (total_erasers : ℕ) :
  number_of_friends = 48 →
  erasers_per_friend = 80 →
  total_erasers = number_of_friends * erasers_per_friend →
  total_erasers = 3840 :=
by
  intros h_friends h_erasers h_total
  sorry

end NUMINAMATH_GPT_diana_erasers_l946_94639


namespace NUMINAMATH_GPT_factor_expression_l946_94627

theorem factor_expression (x : ℝ) : 
  75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l946_94627


namespace NUMINAMATH_GPT_ordered_pair_exists_l946_94641

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_exists_l946_94641


namespace NUMINAMATH_GPT_additional_plates_correct_l946_94655

-- Define the conditions
def original_set_1 : Finset Char := {'B', 'F', 'J', 'N', 'T'}
def original_set_2 : Finset Char := {'E', 'U'}
def original_set_3 : Finset Char := {'G', 'K', 'R', 'Z'}

-- Define the sizes of the original sets
def size_set_1 := (original_set_1.card : Nat) -- 5
def size_set_2 := (original_set_2.card : Nat) -- 2
def size_set_3 := (original_set_3.card : Nat) -- 4

-- Sizes after adding new letters
def new_size_set_1 := size_set_1 + 1 -- 6
def new_size_set_2 := size_set_2 + 1 -- 3
def new_size_set_3 := size_set_3 + 1 -- 5

-- Calculate the original and new number of plates
def original_plates : Nat := size_set_1 * size_set_2 * size_set_3 -- 5 * 2 * 4 = 40
def new_plates : Nat := new_size_set_1 * new_size_set_2 * new_size_set_3 -- 6 * 3 * 5 = 90

-- Calculate the additional plates
def additional_plates : Nat := new_plates - original_plates -- 90 - 40 = 50

-- The proof statement
theorem additional_plates_correct : additional_plates = 50 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_additional_plates_correct_l946_94655


namespace NUMINAMATH_GPT_number_of_bottles_poured_l946_94651

/-- Definition of full cylinder capacity (fixed as 80 bottles) --/
def full_capacity : ℕ := 80

/-- Initial fraction of full capacity --/
def initial_fraction : ℚ := 3 / 4

/-- Final fraction of full capacity --/
def final_fraction : ℚ := 4 / 5

/-- Proof problem: Prove the number of bottles of oil poured into the cylinder --/
theorem number_of_bottles_poured :
  (final_fraction * full_capacity) - (initial_fraction * full_capacity) = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_bottles_poured_l946_94651


namespace NUMINAMATH_GPT_parabola_chord_constant_l946_94681

noncomputable def calcT (x₁ x₂ c : ℝ) : ℝ :=
  let a := x₁^2 + (2*x₁^2 - c)^2
  let b := x₂^2 + (2*x₂^2 - c)^2
  1 / Real.sqrt a + 1 / Real.sqrt b

theorem parabola_chord_constant (c : ℝ) (m x₁ x₂ : ℝ) 
    (h₁ : 2*x₁^2 - m*x₁ - c = 0) 
    (h₂ : 2*x₂^2 - m*x₂ - c = 0) : 
    calcT x₁ x₂ c = -20 / (7 * c) :=
by
  sorry

end NUMINAMATH_GPT_parabola_chord_constant_l946_94681


namespace NUMINAMATH_GPT_percent_defective_units_l946_94664

-- Definition of the given problem conditions
variable (D : ℝ) -- D represents the percentage of defective units

-- The main statement we want to prove
theorem percent_defective_units (h1 : 0.04 * D = 0.36) : D = 9 := by
  sorry

end NUMINAMATH_GPT_percent_defective_units_l946_94664


namespace NUMINAMATH_GPT_sin_sq_sub_sin_double_l946_94631

open Real

theorem sin_sq_sub_sin_double (alpha : ℝ) (h : tan alpha = 1 / 2) : sin alpha ^ 2 - sin (2 * alpha) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_sq_sub_sin_double_l946_94631


namespace NUMINAMATH_GPT_ab_le_one_l946_94677

theorem ab_le_one {a b : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 2) : ab ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_ab_le_one_l946_94677


namespace NUMINAMATH_GPT_max_load_per_truck_l946_94671

-- Definitions based on given conditions
def num_trucks : ℕ := 3
def total_boxes : ℕ := 240
def lighter_box_weight : ℕ := 10
def heavier_box_weight : ℕ := 40

-- Proof problem statement
theorem max_load_per_truck :
  (total_boxes / 2) * lighter_box_weight + (total_boxes / 2) * heavier_box_weight = 6000 →
  6000 / num_trucks = 2000 :=
by sorry

end NUMINAMATH_GPT_max_load_per_truck_l946_94671


namespace NUMINAMATH_GPT_caleb_double_burgers_count_l946_94616

theorem caleb_double_burgers_count
    (S D : ℕ)
    (cost_single cost_double total_hamburgers total_cost : ℝ)
    (h1 : cost_single = 1.00)
    (h2 : cost_double = 1.50)
    (h3 : total_hamburgers = 50)
    (h4 : total_cost = 66.50)
    (h5 : S + D = total_hamburgers)
    (h6 : cost_single * S + cost_double * D = total_cost) :
    D = 33 := 
sorry

end NUMINAMATH_GPT_caleb_double_burgers_count_l946_94616


namespace NUMINAMATH_GPT_garage_sale_total_l946_94640

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end NUMINAMATH_GPT_garage_sale_total_l946_94640


namespace NUMINAMATH_GPT_angle_line_plane_l946_94602

theorem angle_line_plane {l α : Type} (θ : ℝ) (h : θ = 150) : 
  ∃ φ : ℝ, φ = 60 := 
by
  -- This part would require the actual proof.
  sorry

end NUMINAMATH_GPT_angle_line_plane_l946_94602


namespace NUMINAMATH_GPT_trig_identity_l946_94668

theorem trig_identity (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l946_94668


namespace NUMINAMATH_GPT_average_class_score_l946_94629

theorem average_class_score (total_students assigned_day_students make_up_date_students : ℕ)
  (assigned_day_percentage make_up_date_percentage assigned_day_avg_score make_up_date_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 0.70)
  (h3 : make_up_date_percentage = 0.30)
  (h4 : assigned_day_students = 70)
  (h5 : make_up_date_students = 30)
  (h6 : assigned_day_avg_score = 55)
  (h7 : make_up_date_avg_score = 95) :
  (assigned_day_avg_score * assigned_day_students + make_up_date_avg_score * make_up_date_students) / total_students = 67 :=
by
  sorry

end NUMINAMATH_GPT_average_class_score_l946_94629


namespace NUMINAMATH_GPT_no_such_function_exists_l946_94675

theorem no_such_function_exists (f : ℕ → ℕ) (h : ∀ n, f (f n) = n + 2019) : false :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l946_94675


namespace NUMINAMATH_GPT_evaluate_expression_l946_94607

theorem evaluate_expression (x c : ℕ) (h1 : x = 3) (h2 : c = 2) : 
  ((x^2 + c)^2 - (x^2 - c)^2) = 72 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l946_94607


namespace NUMINAMATH_GPT_overlapping_area_fraction_l946_94676

variable (Y X : ℝ)
variable (hY : 0 < Y)
variable (hX : X = (1 / 8) * (2 * Y - X))

theorem overlapping_area_fraction : X = (2 / 9) * Y :=
by
  -- We define the conditions and relationships stated in the problem
  -- Prove the theorem accordingly
  sorry

end NUMINAMATH_GPT_overlapping_area_fraction_l946_94676


namespace NUMINAMATH_GPT_exists_k_for_binary_operation_l946_94670

noncomputable def binary_operation (a b : ℤ) : ℤ := sorry

theorem exists_k_for_binary_operation :
  (∀ (a b c : ℤ), binary_operation a (b + c) = 
      binary_operation b a + binary_operation c a) →
  ∃ (k : ℤ), ∀ (a b : ℤ), binary_operation a b = k * a * b :=
by
  sorry

end NUMINAMATH_GPT_exists_k_for_binary_operation_l946_94670


namespace NUMINAMATH_GPT_find_a_l946_94638
open Real

noncomputable def f (a x : ℝ) := x * sin x + a * x

theorem find_a (a : ℝ) : (deriv (f a) (π / 2) = 1) → a = 0 := by
  sorry

end NUMINAMATH_GPT_find_a_l946_94638


namespace NUMINAMATH_GPT_jame_annual_earnings_difference_l946_94614

-- Define conditions
def new_hourly_wage := 20
def new_hours_per_week := 40
def old_hourly_wage := 16
def old_hours_per_week := 25
def weeks_per_year := 52

-- Define annual earnings calculations
def annual_earnings_old (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

def annual_earnings_new (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

-- Problem statement to prove
theorem jame_annual_earnings_difference :
  annual_earnings_new new_hourly_wage new_hours_per_week weeks_per_year -
  annual_earnings_old old_hourly_wage old_hours_per_week weeks_per_year = 20800 := by
  sorry

end NUMINAMATH_GPT_jame_annual_earnings_difference_l946_94614


namespace NUMINAMATH_GPT_theresa_more_than_thrice_julia_l946_94687

-- Define the problem parameters
variable (tory julia theresa : ℕ)

def tory_videogames : ℕ := 6
def theresa_videogames : ℕ := 11

-- Define the relationships between the numbers of video games
def julia_relationship := julia = tory / 3
def theresa_compared_to_julia := theresa = theresa_videogames
def tory_value := tory = tory_videogames

theorem theresa_more_than_thrice_julia (h1 : julia_relationship tory julia) 
                                       (h2 : tory_value tory)
                                       (h3 : theresa_compared_to_julia theresa) :
  theresa - 3 * julia = 5 :=
by 
  -- Here comes the proof (not required for the task)
  sorry

end NUMINAMATH_GPT_theresa_more_than_thrice_julia_l946_94687


namespace NUMINAMATH_GPT_factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l946_94612

-- Statement for question 1
theorem factorize_m4_minus_5m_plus_4 (m : ℤ) : 
  (m ^ 4 - 5 * m + 4) = (m ^ 4 - 5 * m + 4) := sorry

-- Statement for question 2
theorem factorize_x3_plus_2x2_plus_4x_plus_3 (x : ℝ) :
  (x ^ 3 + 2 * x ^ 2 + 4 * x + 3) = (x + 1) * (x ^ 2 + x + 3) := sorry

-- Statement for question 3
theorem factorize_x5_minus_1 (x : ℝ) :
  (x ^ 5 - 1) = (x - 1) * (x ^ 4 + x ^ 3 + x ^ 2 + x + 1) := sorry

end NUMINAMATH_GPT_factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l946_94612


namespace NUMINAMATH_GPT_average_salary_technicians_correct_l946_94694

section
variable (average_salary_all : ℝ)
variable (total_workers : ℕ)
variable (average_salary_rest : ℝ)
variable (num_technicians : ℕ)

noncomputable def average_salary_technicians
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : ℝ :=
  12000

theorem average_salary_technicians_correct
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : average_salary_technicians average_salary_all total_workers average_salary_rest num_technicians h1 h2 h3 h4 = 12000 :=
sorry

end

end NUMINAMATH_GPT_average_salary_technicians_correct_l946_94694


namespace NUMINAMATH_GPT_geometric_progression_sum_l946_94697

theorem geometric_progression_sum (a q : ℝ) :
  (a + a * q^2 + a * q^4 = 63) →
  (a * q + a * q^3 = 30) →
  (a = 3 ∧ q = 2) ∨ (a = 48 ∧ q = 1 / 2) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_geometric_progression_sum_l946_94697


namespace NUMINAMATH_GPT_bamboo_tube_rice_capacity_l946_94645

theorem bamboo_tube_rice_capacity :
  ∃ (a d : ℝ), 3 * a + 3 * d * (1 + 2) = 4.5 ∧ 
               4 * (a + 5 * d) + 4 * d * (6 + 7 + 8) = 3.8 ∧ 
               (a + 3 * d) + (a + 4 * d) = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_bamboo_tube_rice_capacity_l946_94645


namespace NUMINAMATH_GPT_range_of_k_l946_94646

theorem range_of_k (k : ℝ) : (2 > 0) ∧ (k > 0) ∧ (k < 2) ↔ (0 < k ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l946_94646


namespace NUMINAMATH_GPT_arrange_polynomial_ascending_order_l946_94626

variable {R : Type} [Ring R] (x : R)

def p : R := 3 * x ^ 2 - x + x ^ 3 - 1

theorem arrange_polynomial_ascending_order : 
  p x = -1 - x + 3 * x ^ 2 + x ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_arrange_polynomial_ascending_order_l946_94626


namespace NUMINAMATH_GPT_option_B_proof_option_C_proof_l946_94618

-- Definitions and sequences
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Statement of the problem

theorem option_B_proof (A B : ℝ) :
  (∀ n : ℕ, S n = A * (n : ℝ)^2 + B * n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem option_C_proof :
  (∀ n : ℕ, S n = 1 - (-1)^n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end NUMINAMATH_GPT_option_B_proof_option_C_proof_l946_94618


namespace NUMINAMATH_GPT_red_apples_sold_l946_94650

-- Define the variables and constants
variables (R G : ℕ)

-- Conditions (Definitions)
def ratio_condition : Prop := R / G = 8 / 3
def combine_condition : Prop := R + G = 44

-- Theorem statement to show number of red apples sold is 32 under given conditions
theorem red_apples_sold : ratio_condition R G → combine_condition R G → R = 32 :=
by
sorry

end NUMINAMATH_GPT_red_apples_sold_l946_94650


namespace NUMINAMATH_GPT_instrument_costs_purchasing_plans_l946_94678

variable (x y : ℕ)
variable (a b : ℕ)

theorem instrument_costs : 
  (2 * x + 3 * y = 1700 ∧ 3 * x + y = 1500) →
  x = 400 ∧ y = 300 := 
by 
  intros h
  sorry

theorem purchasing_plans :
  (x = 400) → (y = 300) → (3 * a + 10 = b) →
  (400 * a + 300 * b ≤ 30000) →
  ((760 - 400) * a + (540 - 300) * b ≥ 21600) →
  (a = 18 ∧ b = 64 ∨ a = 19 ∧ b = 67 ∨ a = 20 ∧ b = 70) :=
by
  intros hx hy hab hcost hprofit
  sorry

end NUMINAMATH_GPT_instrument_costs_purchasing_plans_l946_94678


namespace NUMINAMATH_GPT_balance_blue_balls_l946_94690

variable (G Y W B : ℝ)

-- Define the conditions
def condition1 : 4 * G = 8 * B := sorry
def condition2 : 3 * Y = 8 * B := sorry
def condition3 : 4 * B = 3 * W := sorry

-- Prove the required balance of 3G + 4Y + 3W
theorem balance_blue_balls (h1 : 4 * G = 8 * B) (h2 : 3 * Y = 8 * B) (h3 : 4 * B = 3 * W) :
  3 * (2 * B) + 4 * (8 / 3 * B) + 3 * (4 / 3 * B) = 62 / 3 * B := by
  sorry

end NUMINAMATH_GPT_balance_blue_balls_l946_94690


namespace NUMINAMATH_GPT_find_y_given_area_l946_94628

-- Define the problem parameters and conditions
namespace RectangleArea

variables {y : ℝ} (y_pos : y > 0)

-- Define the vertices, they can be expressed but are not required in the statement
def vertices := [(-2, y), (8, y), (-2, 3), (8, 3)]

-- Define the area condition
def area_condition := 10 * (y - 3) = 90

-- Lean statement proving y = 12 given the conditions
theorem find_y_given_area (y_pos : y > 0) (h : 10 * (y - 3) = 90) : y = 12 :=
by
  sorry

end RectangleArea

end NUMINAMATH_GPT_find_y_given_area_l946_94628


namespace NUMINAMATH_GPT_acetone_C_mass_percentage_l946_94649

noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + (1 * atomic_mass_O)
  let total_mass_C := 3 * atomic_mass_C
  (total_mass_C / molar_mass_acetone) * 100

theorem acetone_C_mass_percentage :
  abs (mass_percentage_C_in_acetone - 62.01) < 0.01 := by
  sorry

end NUMINAMATH_GPT_acetone_C_mass_percentage_l946_94649


namespace NUMINAMATH_GPT_xiaobin_duration_l946_94657

def t1 : ℕ := 9
def t2 : ℕ := 15

theorem xiaobin_duration : t2 - t1 = 6 := by
  sorry

end NUMINAMATH_GPT_xiaobin_duration_l946_94657


namespace NUMINAMATH_GPT_train_speed_l946_94667

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ) :
  length_train = 140 →
  length_bridge = 235.03 →
  time_seconds = 30 →
  (length_train + length_bridge) / time_seconds * 3.6 = 45.0036 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_speed_l946_94667


namespace NUMINAMATH_GPT_greatest_value_of_a_greatest_value_of_a_achieved_l946_94688

theorem greatest_value_of_a (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 :=
sorry

theorem greatest_value_of_a_achieved (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120)
  (h2 : Nat.gcd a b = 10) (h3 : 10 ∣ a ∧ 10 ∣ b) (h4 : Nat.lcm a b = 20) : a = 20 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_a_greatest_value_of_a_achieved_l946_94688


namespace NUMINAMATH_GPT_AM_GM_Inequality_four_vars_l946_94693

theorem AM_GM_Inequality_four_vars (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

end NUMINAMATH_GPT_AM_GM_Inequality_four_vars_l946_94693


namespace NUMINAMATH_GPT_students_remaining_after_fifth_stop_l946_94669

theorem students_remaining_after_fifth_stop (initial_students : ℕ) (stops : ℕ) :
  initial_students = 60 →
  stops = 5 →
  (∀ n, (n < stops → ∃ k, n = 3 * k + 1) → ∀ x, x = initial_students * ((2 : ℚ) / 3)^stops) →
  initial_students * ((2 : ℚ) / 3)^stops = (640 / 81 : ℚ) :=
by
  intros h_initial h_stops h_formula
  sorry

end NUMINAMATH_GPT_students_remaining_after_fifth_stop_l946_94669


namespace NUMINAMATH_GPT_initial_games_l946_94680

theorem initial_games (X : ℕ) (h1 : X - 68 + 47 = 74) : X = 95 :=
by
  sorry

end NUMINAMATH_GPT_initial_games_l946_94680


namespace NUMINAMATH_GPT_solution_set_of_inequality_l946_94674

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l946_94674


namespace NUMINAMATH_GPT_voting_proposal_l946_94613

theorem voting_proposal :
  ∀ (T Votes_against Votes_in_favor More_votes_in_favor : ℕ),
    T = 290 →
    Votes_against = (40 * T) / 100 →
    Votes_in_favor = T - Votes_against →
    More_votes_in_favor = Votes_in_favor - Votes_against →
    More_votes_in_favor = 58 :=
by sorry

end NUMINAMATH_GPT_voting_proposal_l946_94613


namespace NUMINAMATH_GPT_cos_segments_ratio_proof_l946_94659

open Real

noncomputable def cos_segments_ratio := 
  let p := 5
  let q := 26
  ∀ x : ℝ, (cos x = cos 50) → (p, q) = (5, 26)

theorem cos_segments_ratio_proof : cos_segments_ratio :=
by 
  sorry

end NUMINAMATH_GPT_cos_segments_ratio_proof_l946_94659


namespace NUMINAMATH_GPT_complex_number_solution_l946_94686

open Complex

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : z = 1 + I :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l946_94686


namespace NUMINAMATH_GPT_min_1x1_tiles_l946_94630

/-- To cover a 23x23 grid using 1x1, 2x2, and 3x3 tiles (without gaps or overlaps),
the minimum number of 1x1 tiles required is 1. -/
theorem min_1x1_tiles (a b c : ℕ) (h : a + 2 * b + 3 * c = 23 * 23) : 
  a ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_1x1_tiles_l946_94630


namespace NUMINAMATH_GPT_prime_solution_l946_94644

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_solution : ∀ (p q : ℕ), 
  is_prime p → is_prime q → 7 * p * q^2 + p = q^3 + 43 * p^3 + 1 → (p = 2 ∧ q = 7) :=
by
  intros p q hp hq h
  sorry

end NUMINAMATH_GPT_prime_solution_l946_94644


namespace NUMINAMATH_GPT_not_algorithm_is_C_l946_94633

-- Definitions based on the conditions recognized in a)
def option_A := "To go from Zhongshan to Beijing, first take a bus, then take a train."
def option_B := "The steps to solve a linear equation are to eliminate the denominator, remove the brackets, transpose terms, combine like terms, and make the coefficient 1."
def option_C := "The equation x^2 - 4x + 3 = 0 has two distinct real roots."
def option_D := "When solving the inequality ax + 3 > 0, the first step is to transpose terms, and the second step is to discuss the sign of a."

-- Problem statement
theorem not_algorithm_is_C : 
  (option_C ≠ "algorithm for solving a problem") ∧ 
  (option_A = "algorithm for solving a problem") ∧ 
  (option_B = "algorithm for solving a problem") ∧ 
  (option_D = "algorithm for solving a problem") :=
  by 
  sorry

end NUMINAMATH_GPT_not_algorithm_is_C_l946_94633


namespace NUMINAMATH_GPT_square_area_25_l946_94647

theorem square_area_25 (side_length : ℝ) (h_side_length : side_length = 5) : side_length * side_length = 25 := 
by
  rw [h_side_length]
  norm_num
  done

end NUMINAMATH_GPT_square_area_25_l946_94647


namespace NUMINAMATH_GPT_clock_hand_swap_times_l946_94603

noncomputable def time_between_2_and_3 : ℚ := (2 * 143 + 370) / 143
noncomputable def time_between_6_and_7 : ℚ := (6 * 143 + 84) / 143

theorem clock_hand_swap_times :
  time_between_2_and_3 = 2 + 31 * 7 / 143 ∧
  time_between_6_and_7 = 6 + 12 * 84 / 143 :=
by
  -- Math proof will go here
  sorry

end NUMINAMATH_GPT_clock_hand_swap_times_l946_94603


namespace NUMINAMATH_GPT_halloween_candy_l946_94660

theorem halloween_candy : 23 - 7 + 21 = 37 :=
by
  sorry

end NUMINAMATH_GPT_halloween_candy_l946_94660


namespace NUMINAMATH_GPT_cost_of_fencing_l946_94625

open Real

theorem cost_of_fencing
  (ratio_length_width : ∃ x : ℝ, 3 * x * 2 * x = 3750)
  (cost_per_meter : ℝ := 0.50) :
  ∃ cost : ℝ, cost = 125 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l946_94625


namespace NUMINAMATH_GPT_katherine_time_20_l946_94672

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end NUMINAMATH_GPT_katherine_time_20_l946_94672


namespace NUMINAMATH_GPT_find_triplets_l946_94643

theorem find_triplets (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a ∣ b + c + 1) (h5 : b ∣ c + a + 1) (h6 : c ∣ a + b + 1) :
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 2) ∨ (a, b, c) = (3, 4, 4) ∨ 
  (a, b, c) = (1, 1, 3) ∨ (a, b, c) = (2, 2, 5) :=
sorry

end NUMINAMATH_GPT_find_triplets_l946_94643


namespace NUMINAMATH_GPT_sequence_term_sequence_sum_l946_94692

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3^(n-1)

def S_n (n : ℕ) : ℕ :=
  (3^n - 1) / 2

theorem sequence_term (n : ℕ) (h : n ≥ 1) :
  a_seq n = 3^(n-1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_term_sequence_sum_l946_94692


namespace NUMINAMATH_GPT_symmetry_axis_of_function_l946_94632

noncomputable def f (varphi : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + varphi)

theorem symmetry_axis_of_function
  (varphi : ℝ) (h1 : |varphi| < Real.pi / 2)
  (h2 : f varphi (Real.pi / 6) = 1) :
  ∃ k : ℤ, (k * Real.pi / 2 + Real.pi / 3 = Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_symmetry_axis_of_function_l946_94632


namespace NUMINAMATH_GPT_Roger_needs_to_delete_20_apps_l946_94642

def max_apps := 50
def recommended_apps := 35
def current_apps := 2 * recommended_apps
def apps_to_delete := current_apps - max_apps

theorem Roger_needs_to_delete_20_apps : apps_to_delete = 20 := by
  sorry

end NUMINAMATH_GPT_Roger_needs_to_delete_20_apps_l946_94642


namespace NUMINAMATH_GPT_colored_pictures_count_l946_94624

def initial_pictures_count : ℕ := 44 + 44
def pictures_left : ℕ := 68

theorem colored_pictures_count : initial_pictures_count - pictures_left = 20 := by
  -- Definitions and proof will go here
  sorry

end NUMINAMATH_GPT_colored_pictures_count_l946_94624
