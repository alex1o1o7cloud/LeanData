import Mathlib

namespace fifth_term_is_67_l4045_404562

def sequence_condition (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) = (s n + s (n + 2)) / 3

theorem fifth_term_is_67 (s : ℕ → ℕ) :
  sequence_condition s →
  s 1 = 3 →
  s 4 = 27 →
  s 5 = 67 := by
  sorry

end fifth_term_is_67_l4045_404562


namespace probability_A_and_B_selected_l4045_404575

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students from 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating the probability of selecting both A and B -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B := by
  sorry


end probability_A_and_B_selected_l4045_404575


namespace problem_statement_l4045_404502

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1) :=
sorry

end problem_statement_l4045_404502


namespace circle_C_equation_chord_AB_length_line_MN_equation_l4045_404530

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l₁
def line_l1 : Set (ℝ × ℝ) := {p | p.1 - p.2 - 2 * Real.sqrt 2 = 0}

-- Define the line l₂
def line_l2 : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 + 5 = 0}

-- Define point G
def point_G : ℝ × ℝ := (1, 3)

-- Statement 1: Equation of circle C
theorem circle_C_equation : circle_C = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by sorry

-- Statement 2: Length of chord AB
theorem chord_AB_length : 
  let chord_AB := circle_C ∩ line_l2
  (Set.ncard chord_AB = 2) → 
  ∃ a b : ℝ × ℝ, a ∈ chord_AB ∧ b ∈ chord_AB ∧ a ≠ b ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 3 := by sorry

-- Statement 3: Equation of line MN
theorem line_MN_equation : 
  ∃ M N : ℝ × ℝ, 
    M ∈ circle_C ∧ N ∈ circle_C ∧ M ≠ N ∧
    ((point_G.1 - M.1)^2 + (point_G.2 - M.2)^2) * 4 = ((M.1)^2 + (M.2)^2) * ((point_G.1)^2 + (point_G.2)^2) ∧
    ((point_G.1 - N.1)^2 + (point_G.2 - N.2)^2) * 4 = ((N.1)^2 + (N.2)^2) * ((point_G.1)^2 + (point_G.2)^2) ∧
    (∀ p : ℝ × ℝ, p ∈ {q | q.1 + 3 * q.2 - 4 = 0} ↔ (p.1 - M.1) * (N.2 - M.2) = (p.2 - M.2) * (N.1 - M.1)) := by sorry

end circle_C_equation_chord_AB_length_line_MN_equation_l4045_404530


namespace inequality_proof_l4045_404578

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end inequality_proof_l4045_404578


namespace M_eq_roster_l4045_404571

def M : Set ℚ := {x | ∃ (m : ℤ) (n : ℕ), n > 0 ∧ n ≤ 3 ∧ abs m < 2 ∧ x = m / n}

theorem M_eq_roster : M = {-1, -1/2, -1/3, 0, 1/3, 1/2, 1} := by sorry

end M_eq_roster_l4045_404571


namespace solar_panel_distribution_l4045_404545

theorem solar_panel_distribution (total_homes : ℕ) (installed_homes : ℕ) (panel_shortage : ℕ) :
  total_homes = 20 →
  installed_homes = 15 →
  panel_shortage = 50 →
  ∃ (panels_per_home : ℕ),
    panels_per_home = 10 ∧
    panels_per_home * total_homes = panels_per_home * installed_homes + panel_shortage :=
by sorry

end solar_panel_distribution_l4045_404545


namespace equation_solution_l4045_404543

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
sorry

end equation_solution_l4045_404543


namespace tom_balloons_l4045_404555

theorem tom_balloons (initial given left : ℕ) : 
  given = 16 → left = 14 → initial = given + left :=
by sorry

end tom_balloons_l4045_404555


namespace raja_income_distribution_l4045_404509

theorem raja_income_distribution (monthly_income : ℝ) 
  (household_percentage : ℝ) (medicine_percentage : ℝ) (savings : ℝ) :
  monthly_income = 37500 →
  household_percentage = 35 →
  medicine_percentage = 5 →
  savings = 15000 →
  ∃ (clothes_percentage : ℝ),
    clothes_percentage = 20 ∧
    (household_percentage / 100 + medicine_percentage / 100 + clothes_percentage / 100) * monthly_income + savings = monthly_income :=
by sorry

end raja_income_distribution_l4045_404509


namespace equation_solution_l4045_404539

theorem equation_solution : ∃ x : ℝ, (3 / (x^2 - 9) + x / (x - 3) = 1) ∧ (x = -4) := by
  sorry

end equation_solution_l4045_404539


namespace longest_segment_in_pie_sector_l4045_404521

theorem longest_segment_in_pie_sector (d : ℝ) (h : d = 12) :
  let r := d / 2
  let sector_angle := 2 * Real.pi / 3
  let chord_length := 2 * r * Real.sin (sector_angle / 2)
  chord_length ^ 2 = 108 := by sorry

end longest_segment_in_pie_sector_l4045_404521


namespace sequence_bound_l4045_404528

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end sequence_bound_l4045_404528


namespace triathlon_running_speed_l4045_404535

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed : ℝ) (average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 4 mph,
    the running speed is 7 mph -/
theorem triathlon_running_speed :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 4
  calculate_running_speed swimming_speed average_speed = 7 := by
sorry

#eval calculate_running_speed 1 4

end triathlon_running_speed_l4045_404535


namespace equal_x_y_l4045_404590

theorem equal_x_y (x y z : ℝ) (h1 : x = 6 - y) (h2 : z^2 = x*y - 9) : x = y := by
  sorry

end equal_x_y_l4045_404590


namespace difference_of_cubes_divisible_by_nine_l4045_404579

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (3*a + 2)^3 - (3*b + 2)^3 = 9*k :=
by sorry

end difference_of_cubes_divisible_by_nine_l4045_404579


namespace total_liquid_consumed_l4045_404564

/-- The amount of cups in one pint -/
def cups_per_pint : ℝ := 2

/-- The amount of cups in one liter -/
def cups_per_liter : ℝ := 4.22675

/-- The amount of pints Elijah drank -/
def elijah_pints : ℝ := 8.5

/-- The amount of pints Emilio drank -/
def emilio_pints : ℝ := 9.5

/-- The amount of liters Isabella drank -/
def isabella_liters : ℝ := 3

/-- The total cups of liquid consumed by Elijah, Emilio, and Isabella -/
def total_cups : ℝ := elijah_pints * cups_per_pint + emilio_pints * cups_per_pint + isabella_liters * cups_per_liter

theorem total_liquid_consumed :
  total_cups = 48.68025 := by sorry

end total_liquid_consumed_l4045_404564


namespace solution_system_equations_l4045_404557

theorem solution_system_equations (x y z : ℝ) : 
  (x^2 - y^2 + z = 27 / (x * y) ∧
   y^2 - z^2 + x = 27 / (y * z) ∧
   z^2 - x^2 + y = 27 / (x * z)) →
  ((x = 3 ∧ y = 3 ∧ z = 3) ∨
   (x = -3 ∧ y = -3 ∧ z = 3) ∨
   (x = -3 ∧ y = 3 ∧ z = -3) ∨
   (x = 3 ∧ y = -3 ∧ z = -3)) :=
by sorry

end solution_system_equations_l4045_404557


namespace smallest_number_divisibility_l4045_404525

theorem smallest_number_divisibility (x : ℕ) : x = 1621432330 ↔ 
  (∀ y : ℕ, y < x → ¬(29 ∣ 5*(y+11) ∧ 53 ∣ 5*(y+11) ∧ 37 ∣ 5*(y+11) ∧ 
                     41 ∣ 5*(y+11) ∧ 47 ∣ 5*(y+11) ∧ 61 ∣ 5*(y+11))) ∧
  (29 ∣ 5*(x+11) ∧ 53 ∣ 5*(x+11) ∧ 37 ∣ 5*(x+11) ∧ 
   41 ∣ 5*(x+11) ∧ 47 ∣ 5*(x+11) ∧ 61 ∣ 5*(x+11)) := by
  sorry

end smallest_number_divisibility_l4045_404525


namespace equation_to_lines_l4045_404585

theorem equation_to_lines (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
sorry

end equation_to_lines_l4045_404585


namespace square_sum_eq_18_l4045_404554

theorem square_sum_eq_18 (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x^2 + y^2 = 18) : 
  x^2 + y^2 = 18 := by
sorry

end square_sum_eq_18_l4045_404554


namespace margin_in_terms_of_selling_price_l4045_404568

/-- Given an article with cost C, selling price S, profit factor p, and margin M,
    prove that M can be expressed in terms of S as (p+n)S / (n(2n + p)). -/
theorem margin_in_terms_of_selling_price
  (C S : ℝ) (p n : ℝ) (h_pos : n > 0)
  (h_margin : ∀ M, M = p * (1/n) * C + C)
  (h_selling : S = C + M) :
  ∃ M, M = (p + n) * S / (n * (2 * n + p)) :=
sorry

end margin_in_terms_of_selling_price_l4045_404568


namespace min_boxes_equal_candies_l4045_404511

/-- The number of candies in a box of "Sweet Mathematics" -/
def SM_box_size : ℕ := 12

/-- The number of candies in a box of "Geometry with Nuts" -/
def GN_box_size : ℕ := 15

/-- The minimum number of boxes of "Sweet Mathematics" needed -/
def min_SM_boxes : ℕ := 5

/-- The minimum number of boxes of "Geometry with Nuts" needed -/
def min_GN_boxes : ℕ := 4

theorem min_boxes_equal_candies :
  min_SM_boxes * SM_box_size = min_GN_boxes * GN_box_size ∧
  ∀ (sm gn : ℕ), sm * SM_box_size = gn * GN_box_size →
    sm ≥ min_SM_boxes ∧ gn ≥ min_GN_boxes :=
by sorry

end min_boxes_equal_candies_l4045_404511


namespace li_zhi_assignment_l4045_404505

-- Define the universities
inductive University
| Tongji
| ShanghaiJiaoTong
| ShanghaiNormal

-- Define the volunteer roles
inductive VolunteerRole
| Translator
| City
| Social

-- Define the students
inductive Student
| LiZhi
| WenWen
| LiuBing

-- Define the assignment function
def assignment (s : Student) : University × VolunteerRole :=
  sorry

-- State the theorem
theorem li_zhi_assignment :
  (∀ s, s = Student.LiZhi → (assignment s).1 ≠ University.Tongji) →
  (∀ s, s = Student.WenWen → (assignment s).1 ≠ University.ShanghaiJiaoTong) →
  (∀ s, (assignment s).1 = University.Tongji → (assignment s).2 ≠ VolunteerRole.Translator) →
  (∀ s, (assignment s).1 = University.ShanghaiJiaoTong → (assignment s).2 = VolunteerRole.City) →
  (∀ s, s = Student.WenWen → (assignment s).2 ≠ VolunteerRole.Social) →
  (assignment Student.LiZhi).1 = University.ShanghaiJiaoTong ∧ 
  (assignment Student.LiZhi).2 = VolunteerRole.City :=
by
  sorry

end li_zhi_assignment_l4045_404505


namespace round_trip_average_speed_river_boat_average_speed_l4045_404510

/-- The average speed for a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : upstream_speed ≠ downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) = 
    (2 * 3 * 7) / (3 + 7) := by
  sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 3 * 7) / (3 + 7) = 4.2 := by
  sorry

end round_trip_average_speed_river_boat_average_speed_l4045_404510


namespace inverse_proportion_problem_l4045_404563

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 40) (h3 : x - y = 10) :
  (7 : ℝ) * (375 / 7) = k := by sorry

end inverse_proportion_problem_l4045_404563


namespace probability_higher_first_lower_second_l4045_404522

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (fun (a, b) => a > b)

theorem probability_higher_first_lower_second :
  (favorable_outcomes card_set).card / (card_set.card * card_set.card : ℚ) = 2 / 5 := by
  sorry

end probability_higher_first_lower_second_l4045_404522


namespace system_solution_l4045_404547

theorem system_solution (x y : ℝ) : 
  (16 * x^3 + 4*x = 16*y + 5) ∧ 
  (16 * y^3 + 4*y = 16*x + 5) → 
  (x = y) ∧ (16 * x^3 - 12*x - 5 = 0) := by
sorry

end system_solution_l4045_404547


namespace orange_packing_problem_l4045_404560

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed to pack all oranges. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges
    when each box holds 10 oranges. -/
theorem orange_packing_problem :
  boxes_needed 2650 10 = 265 := by
  sorry

end orange_packing_problem_l4045_404560


namespace total_weekly_batches_l4045_404567

/-- Represents the types of flour --/
inductive FlourType
| Regular
| GlutenFree
| WholeWheat

/-- Represents a day of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents the flour usage for a single day --/
structure DailyUsage where
  regular : ℕ
  glutenFree : ℕ
  wholeWheat : ℕ
  regularToWholeWheat : ℕ

/-- The number of batches that can be made from one sack of flour --/
def batchesPerSack (t : FlourType) : ℕ :=
  match t with
  | FlourType.Regular => 15
  | FlourType.GlutenFree => 10
  | FlourType.WholeWheat => 12

/-- The conversion rate from regular flour to whole-wheat flour --/
def regularToWholeWheatRate : ℚ := 3/2

/-- The daily flour usage for the week --/
def weekUsage : Day → DailyUsage
| Day.Monday => ⟨4, 3, 2, 0⟩
| Day.Tuesday => ⟨6, 2, 0, 1⟩
| Day.Wednesday => ⟨5, 1, 2, 0⟩
| Day.Thursday => ⟨3, 4, 3, 0⟩
| Day.Friday => ⟨7, 1, 0, 2⟩
| Day.Saturday => ⟨5, 3, 1, 0⟩
| Day.Sunday => ⟨2, 4, 0, 2⟩

/-- Calculates the total number of batches for a given flour type in a week --/
def totalBatches (t : FlourType) : ℕ := sorry

/-- The main theorem: Bruce can make 846 batches of pizza dough in a week --/
theorem total_weekly_batches : (totalBatches FlourType.Regular) + 
                               (totalBatches FlourType.GlutenFree) + 
                               (totalBatches FlourType.WholeWheat) = 846 := sorry

end total_weekly_batches_l4045_404567


namespace negation_equivalence_l4045_404556

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 + Real.sin x < 0)) ↔ (∀ x : ℝ, x > 0 → x^2 + Real.sin x ≥ 0) :=
by sorry

end negation_equivalence_l4045_404556


namespace math_class_grade_distribution_l4045_404506

theorem math_class_grade_distribution (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) : 
  total_students = 40 →
  prob_A = 0.8 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  ∃ (num_B : ℕ), num_B = 13 ∧ 
    (↑num_B : ℚ) * prob_B = (total_students : ℚ) * prob_B := by
  sorry

end math_class_grade_distribution_l4045_404506


namespace partial_fraction_decomposition_product_l4045_404573

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 + 5*x - 14) / (x^3 - 3*x^2 - x + 3) = 
    A / (x - 1) + B / (x - 3) + C / (x + 1) →
  A * B * C = -25 / 2 := by
  sorry

end partial_fraction_decomposition_product_l4045_404573


namespace arithmetic_geometric_mean_ratio_l4045_404598

theorem arithmetic_geometric_mean_ratio (a b m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hm : (a + b) / 2 = m * Real.sqrt (a * b)) : 
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) := by
sorry

end arithmetic_geometric_mean_ratio_l4045_404598


namespace cosine_identity_l4045_404591

theorem cosine_identity (θ : ℝ) 
  (h : Real.cos (π / 4 - θ) = Real.sqrt 3 / 3) : 
  Real.cos (3 * π / 4 + θ) - Real.sin (θ - π / 4) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end cosine_identity_l4045_404591


namespace circular_film_radius_l4045_404558

/-- The radius of a circular film formed by pouring a cylindrical container of liquid onto water -/
theorem circular_film_radius 
  (h : ℝ) -- height of the cylindrical container
  (d : ℝ) -- diameter of the cylindrical container
  (t : ℝ) -- thickness of the resulting circular film
  (h_pos : h > 0)
  (d_pos : d > 0)
  (t_pos : t > 0)
  (h_val : h = 10)
  (d_val : d = 5)
  (t_val : t = 0.2) :
  ∃ (r : ℝ), r^2 = 312.5 ∧ π * (d/2)^2 * h = π * r^2 * t :=
by sorry

end circular_film_radius_l4045_404558


namespace cube_root_of_sqrt_64_l4045_404597

theorem cube_root_of_sqrt_64 : (64 : ℝ) ^ (1/2 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end cube_root_of_sqrt_64_l4045_404597


namespace tens_digit_of_7_pow_35_l4045_404524

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_7_pow_35 : tens_digit (7^35) = 4 := by
  sorry

end tens_digit_of_7_pow_35_l4045_404524


namespace four_students_three_teams_l4045_404534

/-- The number of ways students can sign up for sports teams -/
def signup_ways (num_students : ℕ) (num_teams : ℕ) : ℕ :=
  num_teams ^ num_students

/-- Theorem: 4 students signing up for 3 teams results in 3^4 ways -/
theorem four_students_three_teams :
  signup_ways 4 3 = 3^4 := by
  sorry

end four_students_three_teams_l4045_404534


namespace inverse_g_inverse_14_l4045_404548

def g (x : ℝ) : ℝ := 3 * x - 4

theorem inverse_g_inverse_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 10 / 3 := by sorry

end inverse_g_inverse_14_l4045_404548


namespace intersection_slope_l4045_404582

/-- Given two lines p and q that intersect at (-3, -9), prove that the slope of line q is 0 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 3 → y = k*x - 9 → x = -3 ∧ y = -9) → k = 0 := by
  sorry

end intersection_slope_l4045_404582


namespace fraction_simplification_l4045_404546

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 4) = 3 / 5 := by
  sorry

end fraction_simplification_l4045_404546


namespace zeros_in_Q_l4045_404518

/-- R_k represents an integer whose base-ten representation consists of k consecutive ones -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- Q is the quotient of R_30 and R_6 -/
def Q : ℕ := R 30 / R 6

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 25 := by sorry

end zeros_in_Q_l4045_404518


namespace original_cost_price_l4045_404592

/-- Given an article with a 15% markup and 20% discount, 
    prove that the original cost is 540 when sold at 496.80 --/
theorem original_cost_price (marked_up_price : ℝ) (selling_price : ℝ) : 
  marked_up_price = 1.15 * 540 ∧ 
  selling_price = 0.8 * marked_up_price ∧
  selling_price = 496.80 → 
  540 = (496.80 : ℝ) / 0.92 := by
sorry

#eval (496.80 : Float) / 0.92

end original_cost_price_l4045_404592


namespace new_average_production_l4045_404565

/-- Given the following conditions:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 90 units.
    3. The value of n is 19 days.
    Prove that the new average daily production is 52 units per day. -/
theorem new_average_production (n : ℕ) (prev_avg : ℝ) (today_prod : ℝ) :
  n = 19 ∧ prev_avg = 50 ∧ today_prod = 90 →
  (n * prev_avg + today_prod) / (n + 1) = 52 := by
  sorry

#check new_average_production

end new_average_production_l4045_404565


namespace integer_solutions_of_equation_l4045_404519

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, y^5 + 2*x*y = x^2 + 2*y^4 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 2) := by
  sorry

end integer_solutions_of_equation_l4045_404519


namespace range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l4045_404537

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1: Range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2: Range of a for which ¬p is sufficient but not necessary for ¬q
theorem range_of_a_for_not_p_sufficient_not_necessary (a : ℝ) :
  (∀ x, ¬(p x a) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x a) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l4045_404537


namespace geometric_sequence_product_l4045_404515

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end geometric_sequence_product_l4045_404515


namespace parabola_tangent_point_l4045_404513

theorem parabola_tangent_point (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), (a = -p ∧ b = q) ∧ a^2 = 4*b :=
by sorry

end parabola_tangent_point_l4045_404513


namespace probability_two_red_balls_l4045_404580

def total_balls : ℕ := 15
def red_balls : ℕ := 7
def blue_balls : ℕ := 8

theorem probability_two_red_balls :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end probability_two_red_balls_l4045_404580


namespace prime_fraction_equality_l4045_404541

theorem prime_fraction_equality (A B : ℕ) : 
  Nat.Prime A → 
  Nat.Prime B → 
  A > 0 → 
  B > 0 → 
  (1 : ℚ) / A - (1 : ℚ) / B = 192 / (2005^2 - 2004^2) → 
  B = 211 := by
sorry

end prime_fraction_equality_l4045_404541


namespace drama_club_adult_ticket_price_l4045_404500

/-- Calculates the adult ticket price for a drama club performance --/
theorem drama_club_adult_ticket_price 
  (total_tickets : ℕ) 
  (student_price : ℕ) 
  (total_amount : ℕ) 
  (student_count : ℕ) 
  (h1 : total_tickets = 1500)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_price : ℕ), 
    (total_tickets - student_count) * adult_price + student_count * student_price = total_amount ∧ 
    adult_price = 12 := by
  sorry

end drama_club_adult_ticket_price_l4045_404500


namespace fraction_sum_equality_l4045_404596

theorem fraction_sum_equality : 
  (-3 : ℚ) / 20 + 5 / 200 - 7 / 2000 * 2 = -132 / 1000 := by sorry

end fraction_sum_equality_l4045_404596


namespace expand_expression_l4045_404570

theorem expand_expression (x y : ℝ) : (10 * x - 6 * y + 9) * (3 * y) = 30 * x * y - 18 * y^2 + 27 * y := by
  sorry

end expand_expression_l4045_404570


namespace fish_crate_weight_l4045_404569

theorem fish_crate_weight (total_weight : ℝ) (cost_per_crate : ℝ) (total_cost : ℝ)
  (h1 : total_weight = 540)
  (h2 : cost_per_crate = 1.5)
  (h3 : total_cost = 27) :
  total_weight / (total_cost / cost_per_crate) = 30 :=
by
  sorry

end fish_crate_weight_l4045_404569


namespace molar_mass_calculation_l4045_404595

/-- Given a chemical compound where 3 moles weigh 168 grams, prove that its molar mass is 56 grams per mole. -/
theorem molar_mass_calculation (mass : ℝ) (moles : ℝ) (h1 : mass = 168) (h2 : moles = 3) :
  mass / moles = 56 := by
  sorry

end molar_mass_calculation_l4045_404595


namespace gym_purchase_theorem_l4045_404536

/-- Cost calculation for Option 1 -/
def costOption1 (x : ℕ) : ℚ :=
  1500 + 15 * (x - 20)

/-- Cost calculation for Option 2 -/
def costOption2 (x : ℕ) : ℚ :=
  (1500 + 15 * x) * (9/10)

/-- Cost calculation for the most cost-effective option -/
def costEffectiveOption (x : ℕ) : ℚ :=
  1500 + (x - 20) * 15 * (9/10)

theorem gym_purchase_theorem (x : ℕ) (h : x > 20) :
  (costOption1 40 < costOption2 40) ∧
  (costOption1 100 = costOption2 100) ∧
  (costEffectiveOption 40 < min (costOption1 40) (costOption2 40)) :=
by sorry

end gym_purchase_theorem_l4045_404536


namespace prob_even_sum_three_dice_l4045_404551

/-- The number of faces on each die -/
def num_faces : ℕ := 9

/-- The probability of rolling an even number on one die -/
def p_even : ℚ := 5/9

/-- The probability of rolling an odd number on one die -/
def p_odd : ℚ := 4/9

/-- The probability of getting an even sum when rolling three 9-sided dice -/
theorem prob_even_sum_three_dice : 
  (p_even^3) + 3 * (p_odd^2 * p_even) + 3 * (p_odd * p_even^2) = 665/729 := by
  sorry

end prob_even_sum_three_dice_l4045_404551


namespace power_multiplication_l4045_404540

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_multiplication_l4045_404540


namespace partial_fraction_decomposition_product_l4045_404512

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - 4*x^2 + x + 6) = 
  A / (x - 3) + B / (x + 1) + C / (x - 2) →
  A * B * C = 0 := by
sorry

end partial_fraction_decomposition_product_l4045_404512


namespace election_votes_l4045_404577

theorem election_votes (marcy barry joey : ℕ) : 
  marcy = 3 * barry → 
  barry = 2 * (joey + 3) → 
  marcy = 66 → 
  joey = 8 := by
sorry

end election_votes_l4045_404577


namespace quadratic_equations_intersection_l4045_404516

theorem quadratic_equations_intersection (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x : ℝ, x ∈ M ↔ x^2 - p*x + 6 = 0) ∧
    (∀ x : ℝ, x ∈ N ↔ x^2 + 6*x - q = 0) ∧
    (M ∩ N = {2})) →
  p + q = 21 := by
sorry

end quadratic_equations_intersection_l4045_404516


namespace square_plot_area_l4045_404584

theorem square_plot_area (perimeter : ℝ) (h1 : perimeter * 55 = 3740) : 
  (perimeter / 4) ^ 2 = 289 := by
  sorry

end square_plot_area_l4045_404584


namespace function_inequality_implies_a_range_l4045_404517

/-- Given f(x) = (2x-1)e^x - a(x^2+x) and g(x) = -ax^2 - a, where a ∈ ℝ,
    if f(x) ≥ g(x) for all x ∈ ℝ, then 1 ≤ a ≤ 4e^(3/2) -/
theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, (2*x - 1) * Real.exp x - a*(x^2 + x) ≥ -a*x^2 - a) →
  1 ≤ a ∧ a ≤ 4 * Real.exp (3/2) :=
by sorry

end function_inequality_implies_a_range_l4045_404517


namespace quadrilateral_area_l4045_404549

/-- Pick's Theorem for quadrilaterals -/
def area_by_picks_theorem (interior_points : ℕ) (boundary_points : ℕ) : ℚ :=
  interior_points + boundary_points / 2 - 1

/-- The quadrilateral in the problem -/
structure Quadrilateral where
  interior_points : ℕ
  boundary_points : ℕ

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral where
  interior_points := 12
  boundary_points := 6

theorem quadrilateral_area :
  area_by_picks_theorem problem_quadrilateral.interior_points problem_quadrilateral.boundary_points = 14 := by
  sorry

end quadrilateral_area_l4045_404549


namespace contrapositive_is_true_l4045_404576

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  (x = 2 ∧ y = 3) → (x + y = 5)

-- Define the contrapositive of the original proposition
def contrapositive (x y : ℝ) : Prop :=
  (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3)

-- Theorem stating that the contrapositive is true
theorem contrapositive_is_true : ∀ x y : ℝ, contrapositive x y :=
by
  sorry

end contrapositive_is_true_l4045_404576


namespace line_tangent_to_circle_l4045_404533

/-- The line equation kx - y - 2k + 3 = 0 is tangent to the circle x^2 + (y + 1)^2 = 4 if and only if k = 3/4 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) ↔ k = 3/4 := by
  sorry

end line_tangent_to_circle_l4045_404533


namespace calculation_proof_l4045_404566

theorem calculation_proof : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end calculation_proof_l4045_404566


namespace all_figures_on_page_20_only_figures_in_figure5_on_page_20_l4045_404594

/-- Represents a geometric figure in the book --/
structure GeometricFigure where
  page : Nat

/-- Represents the collection of figures shown in Figure 5 --/
def Figure5 : Set GeometricFigure := sorry

/-- The property that distinguishes the figures in Figure 5 --/
def DistinguishingProperty (f : GeometricFigure) : Prop :=
  f.page = 20

/-- Theorem stating that all figures in Figure 5 have the distinguishing property --/
theorem all_figures_on_page_20 :
  ∀ f ∈ Figure5, DistinguishingProperty f :=
sorry

/-- Theorem stating that no other figures have this property --/
theorem only_figures_in_figure5_on_page_20 :
  ∀ f : GeometricFigure, DistinguishingProperty f → f ∈ Figure5 :=
sorry

end all_figures_on_page_20_only_figures_in_figure5_on_page_20_l4045_404594


namespace gmat_test_problem_l4045_404507

theorem gmat_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.05) :
  p_first + p_second - (1 - p_neither) = 0.55 := by
  sorry

end gmat_test_problem_l4045_404507


namespace zebra_sleeps_longer_l4045_404532

/-- Proves that a zebra sleeps 2 hours more per night than a cougar, given the conditions -/
theorem zebra_sleeps_longer (cougar_sleep : ℕ) (total_sleep : ℕ) : 
  cougar_sleep = 4 →
  total_sleep = 70 →
  (total_sleep - 7 * cougar_sleep) / 7 - cougar_sleep = 2 := by
sorry

end zebra_sleeps_longer_l4045_404532


namespace josie_shopping_shortfall_l4045_404550

def gift_amount : ℕ := 80
def cassette_price : ℕ := 15
def num_cassettes : ℕ := 3
def headphone_price : ℕ := 40
def vinyl_price : ℕ := 12

theorem josie_shopping_shortfall :
  gift_amount < cassette_price * num_cassettes + headphone_price + vinyl_price ∧
  cassette_price * num_cassettes + headphone_price + vinyl_price - gift_amount = 17 :=
by sorry

end josie_shopping_shortfall_l4045_404550


namespace not_perfect_square_2005_l4045_404508

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluates a polynomial at a given point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  sorry

theorem not_perfect_square_2005 (P : IntPolynomial) :
  eval P 5 = 2005 → ¬(is_perfect_square (eval P 2005)) :=
sorry

end not_perfect_square_2005_l4045_404508


namespace little_red_riding_hood_waffles_l4045_404561

theorem little_red_riding_hood_waffles (initial_waffles : ℕ) : 
  (∃ (x : ℕ), 
    initial_waffles = 14 * x ∧ 
    (initial_waffles / 2 - x) / 2 - x = x ∧
    x > 0) →
  initial_waffles % 7 = 0 :=
sorry

end little_red_riding_hood_waffles_l4045_404561


namespace last_two_nonzero_digits_80_factorial_l4045_404572

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the last two nonzero digits
def lastTwoNonzeroDigits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits (factorial 80) = 76 := by
  sorry


end last_two_nonzero_digits_80_factorial_l4045_404572


namespace negation_of_all_rectangles_equal_diagonals_l4045_404553

-- Define a type for rectangles
variable (Rectangle : Type)

-- Define a predicate for equal diagonals
variable (has_equal_diagonals : Rectangle → Prop)

-- Statement to prove
theorem negation_of_all_rectangles_equal_diagonals :
  (¬ ∀ r : Rectangle, has_equal_diagonals r) ↔ (∃ r : Rectangle, ¬ has_equal_diagonals r) := by
  sorry

end negation_of_all_rectangles_equal_diagonals_l4045_404553


namespace third_participant_score_l4045_404586

/-- Represents the score of a participant -/
structure ParticipantScore where
  score : ℕ

/-- Represents the total number of competitions -/
def totalCompetitions : ℕ := 10

/-- Represents the total points awarded in each competition -/
def pointsPerCompetition : ℕ := 4

/-- Theorem: Given the conditions of the competition and two participants' scores,
    the third participant's score is determined -/
theorem third_participant_score 
  (dima misha : ParticipantScore)
  (h1 : dima.score = 22)
  (h2 : misha.score = 8) :
  ∃ yura : ParticipantScore, yura.score = 10 := by
  sorry

end third_participant_score_l4045_404586


namespace keith_pears_given_away_l4045_404542

/-- The number of pears Keith gave away -/
def pears_given_away (keith_initial : ℕ) (mike_initial : ℕ) (remaining : ℕ) : ℕ :=
  keith_initial + mike_initial - remaining

theorem keith_pears_given_away :
  pears_given_away 47 12 13 = 46 := by
  sorry

end keith_pears_given_away_l4045_404542


namespace cube_root_inequality_l4045_404523

theorem cube_root_inequality (x : ℝ) : 
  (x ^ (1/3) : ℝ) - 3 / ((x ^ (1/3) : ℝ) + 4) ≤ 0 ↔ -27 < x ∧ x < -1 := by sorry

end cube_root_inequality_l4045_404523


namespace milk_problem_l4045_404588

theorem milk_problem (initial_milk : ℚ) (given_milk : ℚ) (result : ℚ) : 
  initial_milk = 4 →
  given_milk = 16/3 →
  result = initial_milk - given_milk →
  result = -4/3 :=
by sorry

end milk_problem_l4045_404588


namespace log_stack_sum_l4045_404589

theorem log_stack_sum : ∀ (a l : ℕ) (d : ℤ),
  a = 15 ∧ l = 4 ∧ d = -1 →
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1 : ℤ) * d ∧
  (n : ℤ) * (a + l) / 2 = 114 :=
by sorry

end log_stack_sum_l4045_404589


namespace cone_height_from_cube_l4045_404520

/-- The height of a cone formed by melting a cube -/
theorem cone_height_from_cube (cube_edge : ℝ) (cone_base_area : ℝ) (cone_height : ℝ) : 
  cube_edge = 6 →
  cone_base_area = 54 →
  (cube_edge ^ 3) = (1 / 3) * cone_base_area * cone_height →
  cone_height = 12 :=
by
  sorry

end cone_height_from_cube_l4045_404520


namespace geometric_sequence_sixth_term_l4045_404503

/-- Given a geometric sequence where the first term is 1000 and the eighth term is 125,
    prove that the sixth term is 31.25. -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1000)  -- First term is 1000
  (h2 : a 8 = 125)   -- Eighth term is 125
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 6 = 31.25 := by
sorry

end geometric_sequence_sixth_term_l4045_404503


namespace initial_water_ratio_l4045_404529

/-- Represents the tank and its properties --/
structure Tank where
  capacity : ℝ
  inflow_rate : ℝ
  outflow_rate1 : ℝ
  outflow_rate2 : ℝ
  fill_time : ℝ

/-- Calculates the net flow rate into the tank --/
def net_flow_rate (t : Tank) : ℝ :=
  t.inflow_rate - (t.outflow_rate1 + t.outflow_rate2)

/-- Calculates the initial amount of water in the tank --/
def initial_water (t : Tank) : ℝ :=
  t.capacity - (net_flow_rate t * t.fill_time)

/-- Theorem: The ratio of initial water to total capacity is 1:2 --/
theorem initial_water_ratio (t : Tank) 
  (h1 : t.capacity = 2)
  (h2 : t.inflow_rate = 0.5)
  (h3 : t.outflow_rate1 = 0.25)
  (h4 : t.outflow_rate2 = 1/6)
  (h5 : t.fill_time = 12) :
  initial_water t / t.capacity = 1/2 := by
  sorry

end initial_water_ratio_l4045_404529


namespace common_chord_equation_l4045_404581

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x - 8 = 0) ∧ (x^2 + y^2 + 2*x - 4*y - 4 = 0) →
  (x - y + 1 = 0) :=
by sorry

end common_chord_equation_l4045_404581


namespace room_expansion_proof_l4045_404514

theorem room_expansion_proof (initial_length initial_width increase : ℝ)
  (h1 : initial_length = 13)
  (h2 : initial_width = 18)
  (h3 : increase = 2) :
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  let total_area := 4 * single_room_area + 2 * single_room_area
  total_area = 1800 := by sorry

end room_expansion_proof_l4045_404514


namespace diff_of_squares_equals_fifth_power_l4045_404538

theorem diff_of_squares_equals_fifth_power (a : ℤ) :
  ∃ x y : ℤ, x^2 - y^2 = a^5 := by
sorry

end diff_of_squares_equals_fifth_power_l4045_404538


namespace triangle_altitude_and_median_l4045_404583

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 8)

-- Define the altitude equation
def altitude_eq (x y : ℝ) : Prop := 6 * x - y - 24 = 0

-- Define the median equation
def median_eq (x y : ℝ) : Prop := y = -15/2 * x + 30

-- Theorem statement
theorem triangle_altitude_and_median :
  (∀ x y : ℝ, altitude_eq x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1) ∧ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0) ∧
  (∀ x y : ℝ, median_eq x y ↔ 
    2 * (y - A.2) * (B.1 - C.1) = (x - A.1) * (B.2 + C.2 - 2 * A.2)) :=
sorry

end triangle_altitude_and_median_l4045_404583


namespace hyperbola_equation_l4045_404587

/-- Definition of the hyperbola with given foci and passing point -/
def Hyperbola (f : ℝ × ℝ) (p : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | 
    |((x - f.1)^2 + (y - f.2)^2).sqrt - ((x - f.1)^2 + (y + f.2)^2).sqrt| = 
    |((p.1 - f.1)^2 + (p.2 - f.2)^2).sqrt - ((p.1 - f.1)^2 + (p.2 + f.2)^2).sqrt|}

/-- The theorem stating the equation of the hyperbola -/
theorem hyperbola_equation :
  let f : ℝ × ℝ := (0, 3)
  let p : ℝ × ℝ := (Real.sqrt 15, 4)
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola f p ↔ y^2 / 4 - x^2 / 5 = 1 := by
  sorry

end hyperbola_equation_l4045_404587


namespace min_value_expression_l4045_404504

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 3 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + y^2 + 16 / x^2 + 4 * y / x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + b₀^2 + 16 / a₀^2 + 4 * b₀ / a₀ = min :=
by
  sorry

end min_value_expression_l4045_404504


namespace assign_roles_specific_scenario_l4045_404527

/-- Represents the number of ways to assign roles in a play. -/
def assignRoles (maleRoles femaleRoles eitherRoles maleActors femaleActors : ℕ) : ℕ :=
  (maleActors.descFactorial maleRoles) *
  (femaleActors.descFactorial femaleRoles) *
  ((maleActors + femaleActors - maleRoles - femaleRoles).descFactorial eitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_scenario :
  assignRoles 2 2 3 4 5 = 14400 := by
  sorry

end assign_roles_specific_scenario_l4045_404527


namespace serenas_mother_age_l4045_404544

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_future : ℕ := 6

/-- Serena's mother's age now -/
def mother_age : ℕ := 39

/-- Theorem stating that Serena's mother's current age is 39 -/
theorem serenas_mother_age : 
  (mother_age + years_future) = 3 * (serena_age + years_future) → 
  mother_age = 39 := by
  sorry

end serenas_mother_age_l4045_404544


namespace truncated_cube_volume_ratio_l4045_404593

/-- A convex polyhedron with specific properties -/
structure TruncatedCube where
  /-- The polyhedron has 6 square faces -/
  square_faces : Nat
  /-- The polyhedron has 8 equilateral triangle faces -/
  triangle_faces : Nat
  /-- Each edge is shared between one triangle and one square -/
  shared_edges : Bool
  /-- All dihedral angles between triangles and squares are equal -/
  equal_dihedral_angles : Bool
  /-- The polyhedron can be circumscribed by a sphere -/
  circumscribable : Bool
  /-- Properties of the truncated cube -/
  h_square_faces : square_faces = 6
  h_triangle_faces : triangle_faces = 8
  h_shared_edges : shared_edges = true
  h_equal_dihedral_angles : equal_dihedral_angles = true
  h_circumscribable : circumscribable = true

/-- The theorem stating the ratio of squared volumes -/
theorem truncated_cube_volume_ratio (tc : TruncatedCube) :
  ∃ (v_polyhedron v_sphere : ℝ),
    v_polyhedron > 0 ∧ v_sphere > 0 ∧
    (v_polyhedron / v_sphere)^2 = 25 / (8 * Real.pi^2) :=
sorry

end truncated_cube_volume_ratio_l4045_404593


namespace division_multiplication_relation_l4045_404599

theorem division_multiplication_relation (a b c : ℕ) (h : a / b = c) : 
  c * b = a ∧ a / c = b :=
by
  sorry

end division_multiplication_relation_l4045_404599


namespace meaningful_square_root_range_l4045_404531

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 1 / (x - 1)) → x > 1 :=
by sorry

end meaningful_square_root_range_l4045_404531


namespace fair_ride_cost_l4045_404526

theorem fair_ride_cost (total_tickets : ℕ) (spent_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : num_rides = 8)
  (h4 : total_tickets ≥ spent_tickets) :
  (total_tickets - spent_tickets) / num_rides = 7 := by
sorry

end fair_ride_cost_l4045_404526


namespace specific_pairing_probability_l4045_404552

/-- The probability of a specific pairing in a classroom with random pairings -/
theorem specific_pairing_probability 
  (n : ℕ) -- Total number of students
  (h : n = 32) -- Given number of students in the classroom
  : (1 : ℚ) / (n - 1 : ℚ) = 1 / 31 := by
  sorry

#check specific_pairing_probability

end specific_pairing_probability_l4045_404552


namespace lucy_cookie_sales_l4045_404501

theorem lucy_cookie_sales : ∀ (first_round second_round total : ℕ),
  first_round = 34 →
  second_round = 27 →
  total = first_round + second_round →
  total = 61 := by
  sorry

end lucy_cookie_sales_l4045_404501


namespace karen_start_time_l4045_404559

/-- Proves that Karen starts the race 4 minutes late given the specified conditions. -/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_lead : ℝ) : 
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_lead = 4 →
  (tom_distance / tom_speed - (tom_distance + karen_lead) / karen_speed) * 60 = 4 := by
  sorry

end karen_start_time_l4045_404559


namespace lotto_game_minimum_draws_l4045_404574

theorem lotto_game_minimum_draws (n : ℕ) (h : n = 90) : 
  ∃ k : ℕ, k = 49 ∧ 
  (∀ S : Finset ℕ, S.card = k → S ⊆ Finset.range n → 
    ∃ x ∈ S, x % 3 = 0 ∨ x % 5 = 0) ∧
  (∀ m : ℕ, m < k → 
    ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range n ∧ 
    ∀ x ∈ T, x % 3 ≠ 0 ∧ x % 5 ≠ 0) :=
by sorry


end lotto_game_minimum_draws_l4045_404574
