import Mathlib

namespace value_of_M_l2287_228761

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 4025) : M = 5635 :=
sorry

end value_of_M_l2287_228761


namespace units_digit_n_l2287_228756

theorem units_digit_n (m n : ℕ) (h₁ : m * n = 14^8) (hm : m % 10 = 6) : n % 10 = 1 :=
sorry

end units_digit_n_l2287_228756


namespace find_ab_l2287_228797

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l2287_228797


namespace number_of_solutions_l2287_228720

def f (x : ℝ) : ℝ := |1 - 2 * x|

theorem number_of_solutions :
  (∃ n : ℕ, n = 8 ∧ ∀ x ∈ [0,1], f (f (f x)) = (1 / 2) * x) :=
sorry

end number_of_solutions_l2287_228720


namespace speed_ratio_l2287_228770

-- Define the speeds of A and B
variables (v_A v_B : ℝ)

-- Assume the conditions of the problem
axiom h1 : 200 / v_A = 400 / v_B

-- Prove the ratio of the speeds
theorem speed_ratio : v_A / v_B = 1 / 2 :=
by
  sorry

end speed_ratio_l2287_228770


namespace largest_increase_is_2007_2008_l2287_228746

-- Define the number of students each year
def students_2005 : ℕ := 50
def students_2006 : ℕ := 55
def students_2007 : ℕ := 60
def students_2008 : ℕ := 70
def students_2009 : ℕ := 72
def students_2010 : ℕ := 80

-- Define the percentage increase function
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old) : ℚ) / old * 100

-- Define percentage increases for each pair of consecutive years
def increase_2005_2006 := percentage_increase students_2005 students_2006
def increase_2006_2007 := percentage_increase students_2006 students_2007
def increase_2007_2008 := percentage_increase students_2007 students_2008
def increase_2008_2009 := percentage_increase students_2008 students_2009
def increase_2009_2010 := percentage_increase students_2009 students_2010

-- State the theorem
theorem largest_increase_is_2007_2008 :
  (max (max increase_2005_2006 (max increase_2006_2007 increase_2008_2009))
       increase_2009_2010) < increase_2007_2008 := 
by
  -- Add proof steps if necessary.
  sorry

end largest_increase_is_2007_2008_l2287_228746


namespace length_BF_l2287_228733

-- Define the geometrical configuration
structure Point :=
  (x : ℝ) (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 6 4.8
def C := Point.mk 12 0
def D := Point.mk 3 (-6)
def E := Point.mk 3 0
def F := Point.mk 6 0

-- Define given conditions
def AE := (3 : ℝ)
def CE := (9 : ℝ)
def DE := (6 : ℝ)
def AC := AE + CE

theorem length_BF : (BF = (72 / 7 : ℝ)) :=
by
  sorry

end length_BF_l2287_228733


namespace parabola_properties_l2287_228795

-- Define the conditions
def vertex (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ (x : ℝ), f (v.1) ≤ f x

def vertical_axis_of_symmetry (f : ℝ → ℝ) (h : ℝ) : Prop :=
  ∀ (x : ℝ), f x = f (2 * h - x)

def contains_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define f as the given parabola equation
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

-- The main statement to prove
theorem parabola_properties :
  vertex f (3, -2) ∧ vertical_axis_of_symmetry f 3 ∧ contains_point f (6, 16) := sorry

end parabola_properties_l2287_228795


namespace leon_total_payment_l2287_228735

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end leon_total_payment_l2287_228735


namespace books_ratio_l2287_228722

-- Definitions based on the conditions
def Alyssa_books : Nat := 36
def Nancy_books : Nat := 252

-- Statement to prove
theorem books_ratio :
  (Nancy_books / Alyssa_books) = 7 := 
sorry

end books_ratio_l2287_228722


namespace maximize_revenue_l2287_228785

noncomputable def revenue (p : ℝ) : ℝ := 100 * p - 4 * p^2

theorem maximize_revenue : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 20 ∧ (∀ q : ℝ, 0 ≤ q ∧ q ≤ 20 → revenue q ≤ revenue p) ∧ p = 12.5 := by
  sorry

end maximize_revenue_l2287_228785


namespace freshmen_and_sophomores_without_pet_l2287_228736

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end freshmen_and_sophomores_without_pet_l2287_228736


namespace average_marks_is_25_l2287_228766

variable (M P C : ℕ)

def average_math_chemistry (M C : ℕ) : ℕ :=
  (M + C) / 2

theorem average_marks_is_25 (M P C : ℕ) 
  (h₁ : M + P = 30)
  (h₂ : C = P + 20) : 
  average_math_chemistry M C = 25 :=
by
  sorry

end average_marks_is_25_l2287_228766


namespace gcd_and_sum_of_1729_and_867_l2287_228705

-- Given numbers
def a := 1729
def b := 867

-- Define the problem statement
theorem gcd_and_sum_of_1729_and_867 : Nat.gcd a b = 1 ∧ a + b = 2596 := by
  sorry

end gcd_and_sum_of_1729_and_867_l2287_228705


namespace symmetric_point_xOz_l2287_228743

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l2287_228743


namespace cyclic_sum_nonneg_l2287_228777

theorem cyclic_sum_nonneg 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (k : ℝ) (hk1 : 0 ≤ k) (hk2 : k < 2) :
  (a^2 - b * c) / (b^2 + c^2 + k * a^2)
  + (b^2 - c * a) / (c^2 + a^2 + k * b^2)
  + (c^2 - a * b) / (a^2 + b^2 + k * c^2) ≥ 0 :=
sorry

end cyclic_sum_nonneg_l2287_228777


namespace perimeter_eq_28_l2287_228727

theorem perimeter_eq_28 (PQ QR TS TU : ℝ) (h2 : PQ = 4) (h3 : QR = 4) 
(h5 : TS = 8) (h7 : TU = 4) : 
PQ + QR + TS + TS - TU + TU + TU = 28 := by
  sorry

end perimeter_eq_28_l2287_228727


namespace flower_options_l2287_228725

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end flower_options_l2287_228725


namespace compare_log_inequalities_l2287_228755

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem compare_log_inequalities (a x1 x2 : ℝ) 
  (ha_pos : a > 0) (ha_neq_one : a ≠ 1) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (a > 1 → 1 / 2 * (f a x1 + f a x2) ≤ f a ((x1 + x2) / 2)) ∧
  (0 < a ∧ a < 1 → 1 / 2 * (f a x1 + f a x2) ≥ f a ((x1 + x2) / 2)) :=
by { sorry }

end compare_log_inequalities_l2287_228755


namespace projection_correct_l2287_228773

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := ⟨-1, 3, -4⟩

def projection_yOz_plane (P : Point3D) : Point3D :=
  ⟨0, P.y, P.z⟩

theorem projection_correct :
  projection_yOz_plane P = ⟨0, 3, -4⟩ :=
by
  -- The theorem proof is omitted.
  sorry

end projection_correct_l2287_228773


namespace arithmetic_sequence_a4_possible_values_l2287_228788

theorem arithmetic_sequence_a4_possible_values (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 * a 5 = 9)
  (h3 : a 2 = 3) : 
  a 4 = 3 ∨ a 4 = 7 := 
by 
  sorry

end arithmetic_sequence_a4_possible_values_l2287_228788


namespace point_on_hyperbola_l2287_228710

theorem point_on_hyperbola (x y : ℝ) (h_eqn : y = -4 / x) (h_point : x = -2 ∧ y = 2) : x * y = -4 := 
by
  intros
  sorry

end point_on_hyperbola_l2287_228710


namespace shaded_region_area_l2287_228744

theorem shaded_region_area
  (R r : ℝ)
  (h : r^2 = R^2 - 2500)
  : π * (R^2 - r^2) = 2500 * π :=
by
  sorry

end shaded_region_area_l2287_228744


namespace inradius_of_triangle_l2287_228791

/-- Given conditions for the triangle -/
def perimeter : ℝ := 32
def area : ℝ := 40

/-- The theorem to prove the inradius of the triangle -/
theorem inradius_of_triangle (h : area = (r * perimeter) / 2) : r = 2.5 :=
by
  sorry

end inradius_of_triangle_l2287_228791


namespace necessary_not_sufficient_l2287_228730

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end necessary_not_sufficient_l2287_228730


namespace rectangle_area_l2287_228793

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l2287_228793


namespace parallel_vectors_l2287_228783

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (-2, k)

theorem parallel_vectors (k : ℝ) (h : (1, 4) = c k) : k = -8 :=
sorry

end parallel_vectors_l2287_228783


namespace kelly_total_apples_l2287_228713

variable (initial_apples : ℕ) (additional_apples : ℕ)

theorem kelly_total_apples (h1 : initial_apples = 56) (h2 : additional_apples = 49) :
  initial_apples + additional_apples = 105 :=
by
  sorry

end kelly_total_apples_l2287_228713


namespace find_a_and_b_l2287_228767

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the curve equation
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 3 * x^2 + a

-- Main theorem to prove a = -1 and b = 3 given tangency conditions
theorem find_a_and_b 
  (k : ℝ) (a b : ℝ) (tangent_point : ℝ × ℝ)
  (h_tangent : tangent_point = (1, 3))
  (h_line : line k tangent_point.1 = tangent_point.2)
  (h_curve : curve a b tangent_point.1 = tangent_point.2)
  (h_slope : curve_derivative a tangent_point.1 = k) : 
  a = -1 ∧ b = 3 := 
by
  sorry

end find_a_and_b_l2287_228767


namespace find_n_l2287_228789

def C (k : ℕ) : ℕ :=
  if k = 1 then 0
  else (Nat.factors k).eraseDup.foldr (· + ·) 0

theorem find_n (n : ℕ) : 
  (∀ n, (C (2 ^ n + 1) = C n) ↔ n = 3) := 
by
  sorry

end find_n_l2287_228789


namespace incorrect_neg_p_l2287_228784

theorem incorrect_neg_p (p : ∀ x : ℝ, x ≥ 1) : ¬ (∀ x : ℝ, x < 1) :=
sorry

end incorrect_neg_p_l2287_228784


namespace four_digit_multiples_of_5_count_l2287_228701

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l2287_228701


namespace curve_is_line_l2287_228734

def curve_theta (theta : ℝ) : Prop :=
  theta = Real.pi / 4

theorem curve_is_line : curve_theta θ → (curve_type = "line") :=
by
  intros h
  cases h
  -- This is where the proof would go, but we'll use a placeholder for now.
  -- The essence of the proof will show that all points making an angle of π/4 with the x-axis lie on a line.
  exact sorry

end curve_is_line_l2287_228734


namespace right_angle_triangle_probability_l2287_228775

def vertex_count : ℕ := 16
def ways_to_choose_3_points : ℕ := Nat.choose vertex_count 3
def number_of_rectangles : ℕ := 36
def right_angle_triangles_per_rectangle : ℕ := 4
def total_right_angle_triangles : ℕ := number_of_rectangles * right_angle_triangles_per_rectangle
def probability_right_angle_triangle : ℚ := total_right_angle_triangles / ways_to_choose_3_points

theorem right_angle_triangle_probability :
  probability_right_angle_triangle = (9 / 35 : ℚ) := by
  sorry

end right_angle_triangle_probability_l2287_228775


namespace american_summits_more_water_l2287_228729

-- Definitions based on the conditions
def FosterFarmsChickens := 45
def AmericanSummitsWater := 2 * FosterFarmsChickens
def HormelChickens := 3 * FosterFarmsChickens
def BoudinButchersChickens := HormelChickens / 3
def TotalItems := 375
def ItemsByFourCompanies := FosterFarmsChickens + AmericanSummitsWater + HormelChickens + BoudinButchersChickens
def DelMonteWater := TotalItems - ItemsByFourCompanies
def WaterDifference := AmericanSummitsWater - DelMonteWater

theorem american_summits_more_water : WaterDifference = 30 := by
  sorry

end american_summits_more_water_l2287_228729


namespace roots_of_quadratic_l2287_228781

theorem roots_of_quadratic (x : ℝ) : x^2 + x = 0 ↔ (x = 0 ∨ x = -1) :=
by sorry

end roots_of_quadratic_l2287_228781


namespace trains_cross_time_l2287_228750

noncomputable def time_to_cross_trains : ℝ :=
  200 / (89.992800575953935 * (1000 / 3600))

theorem trains_cross_time :
  abs (time_to_cross_trains - 8) < 1e-7 :=
by
  sorry

end trains_cross_time_l2287_228750


namespace solve_for_x_l2287_228737

theorem solve_for_x (x : ℚ) (h : (x - 3) / (x + 2) + (3 * x - 9) / (x - 3) = 2) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l2287_228737


namespace value_of_MN_l2287_228714

theorem value_of_MN (M N : ℝ) (log : ℝ → ℝ → ℝ)
    (h1 : log (M ^ 2) N = log N (M ^ 2))
    (h2 : M ≠ N)
    (h3 : M * N > 0)
    (h4 : M ≠ 1)
    (h5 : N ≠ 1) :
    M * N = N^(1/2) :=
  sorry

end value_of_MN_l2287_228714


namespace janice_homework_time_l2287_228708

variable (H : ℝ)
variable (cleaning_room walk_dog take_trash : ℝ)

-- Conditions from the problem translated directly
def cleaning_room_time : cleaning_room = H / 2 := sorry
def walk_dog_time : walk_dog = H + 5 := sorry
def take_trash_time : take_trash = H / 6 := sorry
def total_time_before_movie : 35 + (H + cleaning_room + walk_dog + take_trash) = 120 := sorry

-- The main theorem to prove
theorem janice_homework_time (H : ℝ)
        (cleaning_room : ℝ := H / 2)
        (walk_dog : ℝ := H + 5)
        (take_trash : ℝ := H / 6) :
    H + cleaning_room + walk_dog + take_trash + 35 = 120 → H = 30 :=
by
  sorry

end janice_homework_time_l2287_228708


namespace carrots_remaining_l2287_228728

theorem carrots_remaining 
  (total_carrots : ℕ)
  (weight_20_carrots : ℕ)
  (removed_carrots : ℕ)
  (avg_weight_remaining : ℕ)
  (avg_weight_removed : ℕ)
  (h1 : total_carrots = 20)
  (h2 : weight_20_carrots = 3640)
  (h3 : removed_carrots = 4)
  (h4 : avg_weight_remaining = 180)
  (h5 : avg_weight_removed = 190) :
  total_carrots - removed_carrots = 16 :=
by 
  -- h1 : 20 carrots in total
  -- h2 : total weight of 20 carrots is 3640 grams
  -- h3 : 4 carrots are removed
  -- h4 : average weight of remaining carrots is 180 grams
  -- h5 : average weight of removed carrots is 190 grams
  sorry

end carrots_remaining_l2287_228728


namespace target_annual_revenue_l2287_228798

-- Given conditions as definitions
def monthly_sales : ℕ := 4000
def additional_sales : ℕ := 1000

-- The proof problem in Lean statement form
theorem target_annual_revenue : (monthly_sales + additional_sales) * 12 = 60000 := by
  sorry

end target_annual_revenue_l2287_228798


namespace combined_tickets_l2287_228786

-- Definitions for the initial conditions
def stuffedTigerPrice : ℝ := 43
def keychainPrice : ℝ := 5.5
def discount1 : ℝ := 0.20 * stuffedTigerPrice
def discountedTigerPrice : ℝ := stuffedTigerPrice - discount1
def ticketsLeftDave : ℝ := 55
def spentDave : ℝ := discountedTigerPrice + keychainPrice
def initialTicketsDave : ℝ := spentDave + ticketsLeftDave

def dinoToyPrice : ℝ := 65
def discount2 : ℝ := 0.15 * dinoToyPrice
def discountedDinoToyPrice : ℝ := dinoToyPrice - discount2
def ticketsLeftAlex : ℝ := 42
def spentAlex : ℝ := discountedDinoToyPrice
def initialTicketsAlex : ℝ := spentAlex + ticketsLeftAlex

-- Lean statement proving the combined number of tickets at the start
theorem combined_tickets {dave_alex_combined : ℝ} 
    (h1 : dave_alex_combined = initialTicketsDave + initialTicketsAlex) : 
    dave_alex_combined = 192.15 := 
by 
    -- Placeholder for the actual proof
    sorry

end combined_tickets_l2287_228786


namespace factory_sample_capacity_l2287_228742

theorem factory_sample_capacity (n : ℕ) (a_ratio b_ratio c_ratio : ℕ) 
  (total_ratio : a_ratio + b_ratio + c_ratio = 10) (a_sample : ℕ)
  (h : a_sample = 16) (h_ratio : a_ratio = 2) :
  n = 80 :=
by
  -- sample calculations proof would normally be here
  sorry

end factory_sample_capacity_l2287_228742


namespace inverse_proposition_false_l2287_228799

theorem inverse_proposition_false (a b c : ℝ) : 
  ¬ (a > b → ((c ≠ 0) ∧ (a / (c * c)) > (b / (c * c))))
:= 
by 
  -- Outline indicating that the proof will follow from checking cases
  sorry

end inverse_proposition_false_l2287_228799


namespace fraction_expression_equiv_l2287_228704

theorem fraction_expression_equiv:
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := 
by 
  sorry

end fraction_expression_equiv_l2287_228704


namespace compare_exponents_l2287_228724

theorem compare_exponents (n : ℕ) (hn : n > 8) :
  let a := Real.sqrt n
  let b := Real.sqrt (n + 1)
  a^b > b^a :=
sorry

end compare_exponents_l2287_228724


namespace mail_difference_eq_15_l2287_228702

variable (Monday Tuesday Wednesday Thursday : ℕ)
variable (total : ℕ)

theorem mail_difference_eq_15
  (h1 : Monday = 65)
  (h2 : Tuesday = Monday + 10)
  (h3 : Wednesday = Tuesday - 5)
  (h4 : total = 295)
  (h5 : total = Monday + Tuesday + Wednesday + Thursday) :
  Thursday - Wednesday = 15 := 
  by
  sorry

end mail_difference_eq_15_l2287_228702


namespace election_win_percentage_l2287_228776

theorem election_win_percentage (total_votes : ℕ) (james_percentage : ℝ) (additional_votes_needed : ℕ) (votes_needed_to_win_percentage : ℝ) :
    total_votes = 2000 →
    james_percentage = 0.005 →
    additional_votes_needed = 991 →
    votes_needed_to_win_percentage = (1001 / 2000) * 100 →
    votes_needed_to_win_percentage > 50.05 :=
by
  intros h_total_votes h_james_percentage h_additional_votes_needed h_votes_needed_to_win_percentage
  sorry

end election_win_percentage_l2287_228776


namespace travel_distance_l2287_228792

theorem travel_distance (x t : ℕ) (h : t = 14400) (h_eq : 12 * x + 12 * (2 * x) = t) : x = 400 :=
by
  sorry

end travel_distance_l2287_228792


namespace point_in_third_quadrant_l2287_228758

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : 
  (-b < 0) ∧ (a < 0) ∧ (-b > a) :=
by
  sorry

end point_in_third_quadrant_l2287_228758


namespace ball_hits_ground_at_10_over_7_l2287_228732

def ball_hits_ground (t : ℚ) : Prop :=
  -4.9 * t^2 + 3.5 * t + 5 = 0

theorem ball_hits_ground_at_10_over_7 : ball_hits_ground (10 / 7) :=
by
  sorry

end ball_hits_ground_at_10_over_7_l2287_228732


namespace count_numbers_1000_to_5000_l2287_228721

def countFourDigitNumbersInRange (lower upper : ℕ) : ℕ :=
  if lower <= upper then upper - lower + 1 else 0

theorem count_numbers_1000_to_5000 : countFourDigitNumbersInRange 1000 5000 = 4001 :=
by
  sorry

end count_numbers_1000_to_5000_l2287_228721


namespace dice_sum_not_22_l2287_228726

theorem dice_sum_not_22 (a b c d e : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 6) (h₁ : 1 ≤ b ∧ b ≤ 6)
  (h₂ : 1 ≤ c ∧ c ≤ 6) (h₃ : 1 ≤ d ∧ d ≤ 6) (h₄ : 1 ≤ e ∧ e ≤ 6) 
  (h₅ : a * b * c * d * e = 432) : a + b + c + d + e ≠ 22 :=
sorry

end dice_sum_not_22_l2287_228726


namespace award_medals_at_most_one_canadian_l2287_228764

/-- Definition of conditions -/
def sprinter_count := 10 -- Total number of sprinters
def canadian_sprinter_count := 4 -- Number of Canadian sprinters
def medals := ["Gold", "Silver", "Bronze"] -- Types of medals

/-- Definition stating the requirement of the problem -/
def atMostOneCanadianMedal (total_sprinters : Nat) (canadian_sprinters : Nat) 
    (medal_types : List String) : Bool := 
  if total_sprinters = sprinter_count ∧ canadian_sprinters = canadian_sprinter_count ∧ medal_types = medals then
    true
  else
    false

/-- Statement to prove the number of ways to award the medals -/
theorem award_medals_at_most_one_canadian :
  (atMostOneCanadianMedal sprinter_count canadian_sprinter_count medals) →
  ∃ (ways : Nat), ways = 480 :=
by
  sorry

end award_medals_at_most_one_canadian_l2287_228764


namespace monthly_rent_calculation_l2287_228716

-- Definitions based on the problem conditions
def investment_amount : ℝ := 20000
def desired_annual_return_rate : ℝ := 0.06
def annual_property_taxes : ℝ := 650
def maintenance_percentage : ℝ := 0.15

-- Theorem stating the mathematically equivalent problem
theorem monthly_rent_calculation : 
  let required_annual_return := desired_annual_return_rate * investment_amount
  let total_annual_earnings := required_annual_return + annual_property_taxes
  let monthly_earnings_target := total_annual_earnings / 12
  let monthly_rent := monthly_earnings_target / (1 - maintenance_percentage)
  monthly_rent = 181.38 :=
by
  sorry

end monthly_rent_calculation_l2287_228716


namespace pyramid_volume_l2287_228787

def area_SAB : ℝ := 9
def area_SBC : ℝ := 9
def area_SCD : ℝ := 27
def area_SDA : ℝ := 27
def area_ABCD : ℝ := 36
def dihedral_angle_equal := ∀ (α β γ δ: ℝ), α = β ∧ β = γ ∧ γ = δ

theorem pyramid_volume (h_eq_dihedral : dihedral_angle_equal)
  (area_conditions : area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27)
  (area_quadrilateral : area_ABCD = 36) :
  (1 / 3 * area_ABCD * 4.5) = 54 :=
sorry

end pyramid_volume_l2287_228787


namespace intersecting_graphs_l2287_228741

theorem intersecting_graphs (a b c d : ℝ) (h₁ : (3, 6) = (3, -|3 - a| + b))
  (h₂ : (9, 2) = (9, -|9 - a| + b))
  (h₃ : (3, 6) = (3, |3 - c| + d))
  (h₄ : (9, 2) = (9, |9 - c| + d)) : 
  a + c = 12 := 
sorry

end intersecting_graphs_l2287_228741


namespace paving_cost_correct_l2287_228700

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 400
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost_correct :
  cost = 8250 := by
  sorry

end paving_cost_correct_l2287_228700


namespace no_such_function_exists_l2287_228763

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f^[n] n = n + 1 :=
by
  sorry

end no_such_function_exists_l2287_228763


namespace point_in_third_quadrant_l2287_228752

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l2287_228752


namespace find_m_l2287_228739

open Real

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (sin x ^ 4 + cos x ^ 4) + m * (sin x + cos x) ^ 4

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x m ≤ 5) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x m = 5) :=
sorry

end find_m_l2287_228739


namespace a_minus_b_is_perfect_square_l2287_228779
-- Import necessary libraries

-- Define the problem in Lean
theorem a_minus_b_is_perfect_square (a b c : ℕ) (h1: Nat.gcd a (Nat.gcd b c) = 1) 
    (h2: (ab : ℚ) / (a - b) = c) : ∃ k : ℕ, a - b = k * k :=
by
  sorry

end a_minus_b_is_perfect_square_l2287_228779


namespace ab_a4_b4_divisible_by_30_l2287_228765

theorem ab_a4_b4_divisible_by_30 (a b : Int) : 30 ∣ a * b * (a^4 - b^4) := 
by
  sorry

end ab_a4_b4_divisible_by_30_l2287_228765


namespace packages_per_truck_l2287_228712

theorem packages_per_truck (total_packages : ℕ) (number_of_trucks : ℕ) (h1 : total_packages = 490) (h2 : number_of_trucks = 7) :
  (total_packages / number_of_trucks) = 70 := by
  sorry

end packages_per_truck_l2287_228712


namespace pencils_per_box_l2287_228790

theorem pencils_per_box (total_pencils : ℝ) (num_boxes : ℝ) (pencils_per_box : ℝ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4.0) 
  (h3 : pencils_per_box = total_pencils / num_boxes) : 
  pencils_per_box = 648 :=
by
  sorry

end pencils_per_box_l2287_228790


namespace probability_at_least_one_white_ball_stall_owner_monthly_earning_l2287_228745

noncomputable def prob_at_least_one_white_ball : ℚ :=
1 - (3 / 10)

theorem probability_at_least_one_white_ball : prob_at_least_one_white_ball = 9 / 10 :=
sorry

noncomputable def expected_monthly_earnings (daily_draws : ℕ) (days_in_month : ℕ) : ℤ :=
(days_in_month * (90 * 1 - 10 * 5))

theorem stall_owner_monthly_earning (daily_draws : ℕ) (days_in_month : ℕ) :
  daily_draws = 100 → days_in_month = 30 →
  expected_monthly_earnings daily_draws days_in_month = 1200 :=
sorry

end probability_at_least_one_white_ball_stall_owner_monthly_earning_l2287_228745


namespace b7_value_l2287_228718

theorem b7_value (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h₀a : a 0 = 3) (h₀b : b 0 = 4)
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 / b n)
  (h₂ : ∀ n, b (n + 1) = b n ^ 2 / a n) :
  b 7 = 4 ^ 730 / 3 ^ 1093 :=
by
  sorry

end b7_value_l2287_228718


namespace divisor_of_number_l2287_228703

theorem divisor_of_number : 
  ∃ D, 
    let x := 75 
    let R' := 7 
    let Q := R' + 8 
    x = D * Q + 0 :=
by
  sorry

end divisor_of_number_l2287_228703


namespace q_can_be_true_or_false_l2287_228759

-- Define the propositions p and q
variables (p q : Prop)

-- The assumptions given in the problem
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ p

-- The statement we want to prove
theorem q_can_be_true_or_false : ∀ q, q ∨ ¬ q :=
by
  intro q
  exact em q -- Use the principle of excluded middle

end q_can_be_true_or_false_l2287_228759


namespace convert_13_to_binary_l2287_228709

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem convert_13_to_binary : decimal_to_binary 13 = [1, 1, 0, 1] :=
  by
    sorry -- Proof to be provided

end convert_13_to_binary_l2287_228709


namespace equal_integers_l2287_228715

theorem equal_integers (a b : ℕ)
  (h : ∀ n : ℕ, n > 0 → a > 0 → b > 0 → (a^n + n) ∣ (b^n + n)) : a = b := 
sorry

end equal_integers_l2287_228715


namespace arithmetic_evaluation_l2287_228762

theorem arithmetic_evaluation :
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end arithmetic_evaluation_l2287_228762


namespace quadratic_real_roots_l2287_228747

theorem quadratic_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ (x^2 - m * x + (m - 1) = 0) ∧ (y^2 - m * y + (m - 1) = 0) 
  ∨ ∃ z : ℝ, (z^2 - m * z + (m - 1) = 0) := 
sorry

end quadratic_real_roots_l2287_228747


namespace fraction_eval_l2287_228774

theorem fraction_eval : 1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 89 :=
by
  sorry

end fraction_eval_l2287_228774


namespace jack_morning_emails_l2287_228772

theorem jack_morning_emails (x : ℕ) (aft_mails eve_mails total_morn_eve : ℕ) (h1: aft_mails = 4) (h2: eve_mails = 8) (h3: total_morn_eve = 11) :
  x = total_morn_eve - eve_mails :=
by 
  sorry

end jack_morning_emails_l2287_228772


namespace factorization_correct_l2287_228769

theorem factorization_correct (x y : ℝ) :
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2) * (x^2 - 3*y + 2) :=
by
  sorry

end factorization_correct_l2287_228769


namespace smallest_number_is_3_l2287_228778

theorem smallest_number_is_3 (a b c : ℝ) (h1 : (a + b + c) / 3 = 7) (h2 : a = 9 ∨ b = 9 ∨ c = 9) : min (min a b) c = 3 := 
sorry

end smallest_number_is_3_l2287_228778


namespace C_neither_necessary_nor_sufficient_for_A_l2287_228782

theorem C_neither_necessary_nor_sufficient_for_A 
  (A B C : Prop) 
  (h1 : B → C)
  (h2 : B → A) : 
  ¬(A → C) ∧ ¬(C → A) :=
by
  sorry

end C_neither_necessary_nor_sufficient_for_A_l2287_228782


namespace odd_function_period_2pi_l2287_228757

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem odd_function_period_2pi (x : ℝ) : 
  f (-x) = -f (x) ∧ 
  ∃ p > 0, p = 2 * Real.pi ∧ ∀ x, f (x + p) = f (x) := 
by
  sorry

end odd_function_period_2pi_l2287_228757


namespace smallest_positive_integer_in_linear_combination_l2287_228738

theorem smallest_positive_integer_in_linear_combination :
  ∃ m n : ℤ, 2016 * m + 43200 * n = 24 :=
by
  sorry

end smallest_positive_integer_in_linear_combination_l2287_228738


namespace inequality_implies_double_l2287_228717

-- Define the condition
variables {x y : ℝ}

theorem inequality_implies_double (h : x < y) : 2 * x < 2 * y :=
  sorry

end inequality_implies_double_l2287_228717


namespace line_equation_l2287_228706

-- Define the structure of a point
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the projection condition
def projection_condition (P : Point) (l : ℤ → ℤ → Prop) : Prop :=
  l P.x P.y ∧ ∀ (Q : Point), l Q.x Q.y → (Q.x ^ 2 + Q.y ^ 2) ≥ (P.x ^ 2 + P.y ^ 2)

-- Define the point P(-2, 1)
def P : Point := ⟨ -2, 1 ⟩

-- Define line l
def line_l (x y : ℤ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem line_equation :
  projection_condition P line_l → ∀ (x y : ℤ), line_l x y ↔ 2 * x - y + 5 = 0 :=
by
  sorry

end line_equation_l2287_228706


namespace find_a_l2287_228740

-- Define the conditions
def parabola_equation (a : ℝ) (x : ℝ) : ℝ := a * x^2
def axis_of_symmetry : ℝ := -2

-- The main theorem: proving the value of a
theorem find_a (a : ℝ) : (axis_of_symmetry = - (1 / (4 * a))) → a = 1/8 :=
by
  intro h
  sorry

end find_a_l2287_228740


namespace polynomial_sum_l2287_228751

def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end polynomial_sum_l2287_228751


namespace train_combined_distance_l2287_228719

/-- Prove that the combined distance covered by three trains is 3480 km,
    given their respective speeds and travel times. -/
theorem train_combined_distance : 
  let speed_A := 150 -- Speed of Train A in km/h
  let time_A := 8     -- Time Train A travels in hours
  let speed_B := 180 -- Speed of Train B in km/h
  let time_B := 6     -- Time Train B travels in hours
  let speed_C := 120 -- Speed of Train C in km/h
  let time_C := 10    -- Time Train C travels in hours
  let distance_A := speed_A * time_A -- Distance covered by Train A
  let distance_B := speed_B * time_B -- Distance covered by Train B
  let distance_C := speed_C * time_C -- Distance covered by Train C
  let combined_distance := distance_A + distance_B + distance_C -- Combined distance covered by all trains
  combined_distance = 3480 :=
by
  sorry

end train_combined_distance_l2287_228719


namespace find_a_pure_imaginary_l2287_228748

theorem find_a_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = (0 : ℝ) + I) :
  (∃ b : ℝ, a - (17 / (4 - i)) = (0 + b*I)) → a = 4 :=
by
  sorry

end find_a_pure_imaginary_l2287_228748


namespace solve_for_m_l2287_228749

-- Define the conditions as hypotheses
def hyperbola_equation (x y : Real) (m : Real) : Prop :=
  (x^2)/(m+9) + (y^2)/9 = 1

def eccentricity (e : Real) (a b : Real) : Prop :=
  e = 2 ∧ e^2 = 1 + (b^2)/(a^2)

-- Prove that m = -36 given the conditions
theorem solve_for_m (m : Real) (h : hyperbola_equation x y m) (h_ecc : eccentricity 2 3 (Real.sqrt (-(m+9)))) :
  m = -36 :=
sorry

end solve_for_m_l2287_228749


namespace primes_solution_l2287_228731

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end primes_solution_l2287_228731


namespace total_legs_correct_l2287_228754

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l2287_228754


namespace sum_of_fractions_to_decimal_l2287_228794

theorem sum_of_fractions_to_decimal :
  ((2 / 40 : ℚ) + (4 / 80) + (6 / 120) + (9 / 180) : ℚ) = 0.2 :=
by
  sorry

end sum_of_fractions_to_decimal_l2287_228794


namespace product_of_solutions_l2287_228780

theorem product_of_solutions : 
  ∀ x : ℝ, 5 = -2 * x^2 + 6 * x → (∃ α β : ℝ, (α ≠ β ∧ (α * β = 5 / 2))) :=
by
  sorry

end product_of_solutions_l2287_228780


namespace smallest_number_condition_l2287_228753

theorem smallest_number_condition
  (x : ℕ)
  (h1 : (x - 24) % 5 = 0)
  (h2 : (x - 24) % 10 = 0)
  (h3 : (x - 24) % 15 = 0)
  (h4 : (x - 24) / 30 = 84)
  : x = 2544 := 
sorry

end smallest_number_condition_l2287_228753


namespace compare_cubics_l2287_228771

variable {a b : ℝ}

theorem compare_cubics (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end compare_cubics_l2287_228771


namespace rectangle_area_ratio_l2287_228768

theorem rectangle_area_ratio (l b : ℕ) (h1 : l = b + 10) (h2 : b = 8) : (l * b) / b = 18 := by
  sorry

end rectangle_area_ratio_l2287_228768


namespace valid_q_values_l2287_228707

theorem valid_q_values (q : ℕ) (h : q > 0) :
  q = 3 ∨ q = 4 ∨ q = 9 ∨ q = 28 ↔ ((5 * q + 40) / (3 * q - 8)) * (3 * q - 8) = 5 * q + 40 :=
by
  sorry

end valid_q_values_l2287_228707


namespace ratio_Rachel_Sara_l2287_228796

-- Define Sara's spending
def Sara_shoes_spending : ℝ := 50
def Sara_dress_spending : ℝ := 200

-- Define Rachel's budget
def Rachel_budget : ℝ := 500

-- Calculate Sara's total spending
def Sara_total_spending : ℝ := Sara_shoes_spending + Sara_dress_spending

-- Define the theorem to prove the ratio
theorem ratio_Rachel_Sara : (Rachel_budget / Sara_total_spending) = 2 := by
  -- Proof is omitted (you would fill in the proof here)
  sorry

end ratio_Rachel_Sara_l2287_228796


namespace tangent_line_at_x_2_increasing_on_1_to_infinity_l2287_228711

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Subpart I
theorem tangent_line_at_x_2 (a b : ℝ) :
  (a / 2 + 2 = 1) ∧ (2 + a * Real.log 2 = 2 + b) → (a = -2 ∧ b = -2 * Real.log 2) :=
by
  sorry

-- Subpart II
theorem increasing_on_1_to_infinity (a : ℝ) :
  (∀ x > 1, (x + a / x) ≥ 0) → (a ≥ -1) :=
by
  sorry

end tangent_line_at_x_2_increasing_on_1_to_infinity_l2287_228711


namespace min_value_fractions_l2287_228723

open Real

theorem min_value_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) :
  3 ≤ (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a)) :=
sorry

end min_value_fractions_l2287_228723


namespace proposition_true_l2287_228760

theorem proposition_true (x y : ℝ) : x + 2 * y ≠ 5 → (x ≠ 1 ∨ y ≠ 2) :=
by
  sorry

end proposition_true_l2287_228760
