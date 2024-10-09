import Mathlib

namespace geometric_sequence_sum_condition_l1645_164554

theorem geometric_sequence_sum_condition
  (a_1 r : ℝ) 
  (S₄ : ℝ := a_1 * (1 + r + r^2 + r^3)) 
  (S₈ : ℝ := S₄ + a_1 * (r^4 + r^5 + r^6 + r^7)) 
  (h₁ : S₄ = 1) 
  (h₂ : S₈ = 3) :
  a_1 * r^16 * (1 + r + r^2 + r^3) = 8 := 
sorry

end geometric_sequence_sum_condition_l1645_164554


namespace domain_sqrt_quot_l1645_164571

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≠ 0}

theorem domain_sqrt_quot (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | x > 0}) :=
by
  sorry

end domain_sqrt_quot_l1645_164571


namespace original_purchase_price_first_commodity_l1645_164599

theorem original_purchase_price_first_commodity (x y : ℝ) 
  (h1 : 1.07 * (x + y) = 827) 
  (h2 : x = y + 127) : 
  x = 450.415 :=
  sorry

end original_purchase_price_first_commodity_l1645_164599


namespace last_two_digits_of_sum_l1645_164598

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l1645_164598


namespace marketing_firm_l1645_164551

variable (Total_households : ℕ) (A_only : ℕ) (A_and_B : ℕ) (B_to_A_and_B_ratio : ℕ)

def neither_soap_households : ℕ :=
  Total_households - (A_only + (B_to_A_and_B_ratio * A_and_B) + A_and_B)

theorem marketing_firm (h1 : Total_households = 300)
                       (h2 : A_only = 60)
                       (h3 : A_and_B = 40)
                       (h4 : B_to_A_and_B_ratio = 3)
                       : neither_soap_households 300 60 40 3 = 80 :=
by {
  sorry
}

end marketing_firm_l1645_164551


namespace corrected_mean_of_observations_l1645_164520

theorem corrected_mean_of_observations (mean : ℝ) (n : ℕ) (incorrect_observation : ℝ) (correct_observation : ℝ) 
  (h_mean : mean = 41) (h_n : n = 50) (h_incorrect_observation : incorrect_observation = 23) (h_correct_observation : correct_observation = 48) 
  (h_sum_incorrect : mean * n = 2050) : 
  (mean * n - incorrect_observation + correct_observation) / n = 41.5 :=
by
  sorry

end corrected_mean_of_observations_l1645_164520


namespace total_distance_A_C_B_l1645_164513

noncomputable section

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A : point := (-3, 5)
def B : point := (5, -3)
def C : point := (0, 0)

theorem total_distance_A_C_B :
  distance A C + distance C B = 2 * sqrt 34 :=
by
  sorry

end total_distance_A_C_B_l1645_164513


namespace parabola_focus_distance_l1645_164535

theorem parabola_focus_distance
  (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 = 3 - p / 2) 
  (h2 : x2 = 2 - p / 2)
  (h3 : y1^2 = 2 * p * x1)
  (h4 : y2^2 = 2 * p * x2)
  (h5 : y1^2 / y2^2 = x1 / x2) : 
  p = 12 / 5 := 
sorry

end parabola_focus_distance_l1645_164535


namespace expand_expression_l1645_164507

variables {R : Type*} [CommRing R] (x : R)

theorem expand_expression : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 :=
by sorry

end expand_expression_l1645_164507


namespace intersection_eq_T_l1645_164505

noncomputable def S : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 3 * x + 2 }
noncomputable def T : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x ^ 2 - 1 }

theorem intersection_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_eq_T_l1645_164505


namespace ray_walks_to_park_l1645_164547

theorem ray_walks_to_park (x : ℤ) (h1 : 3 * (x + 7 + 11) = 66) : x = 4 :=
by
  -- solving steps are skipped
  sorry

end ray_walks_to_park_l1645_164547


namespace inequality_abc_ad_bc_bd_cd_l1645_164577

theorem inequality_abc_ad_bc_bd_cd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) 
  ≤ (3 / 8) * (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 := sorry

end inequality_abc_ad_bc_bd_cd_l1645_164577


namespace maryann_rescue_time_l1645_164530

def time_to_free_cheaph (minutes : ℕ) : ℕ := 6
def time_to_free_expenh (minutes : ℕ) : ℕ := 8
def num_friends : ℕ := 3

theorem maryann_rescue_time : (time_to_free_cheaph 6 + time_to_free_expenh 8) * num_friends = 42 := 
by
  sorry

end maryann_rescue_time_l1645_164530


namespace suzy_final_books_l1645_164523

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l1645_164523


namespace number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l1645_164515

-- Part (a)
theorem number_of_ways_to_choose_4_from_28 :
  (Nat.choose 28 4) = 20475 :=
sorry

-- Part (b)
theorem number_of_ways_to_choose_3_from_27_with_kolya_included :
  (Nat.choose 27 3) = 2925 :=
sorry

end number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l1645_164515


namespace cakes_difference_l1645_164564

theorem cakes_difference :
  let bought := 154
  let sold := 91
  bought - sold = 63 :=
by
  let bought := 154
  let sold := 91
  show bought - sold = 63
  sorry

end cakes_difference_l1645_164564


namespace gcd_60_90_150_l1645_164524

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l1645_164524


namespace noncongruent_triangles_count_l1645_164533

/-- Prove the number of noncongruent integer-sided triangles with positive area,
    perimeter less than 20, that are neither equilateral, isosceles, nor right triangles
    is 17 -/
theorem noncongruent_triangles_count:
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c < 20 ∧ a + b > c ∧ a < b ∧ b < c ∧ 
         ¬(a = b ∨ b = c ∨ a = c) ∧ ¬(a * a + b * b = c * c)) ∧ 
    s.card = 17 := 
sorry

end noncongruent_triangles_count_l1645_164533


namespace four_thirds_of_number_is_36_l1645_164562

theorem four_thirds_of_number_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 :=
  sorry

end four_thirds_of_number_is_36_l1645_164562


namespace maximize_f_at_1_5_l1645_164540

noncomputable def f (x: ℝ) : ℝ := -3 * x^2 + 9 * x + 5

theorem maximize_f_at_1_5 : ∀ x: ℝ, f 1.5 ≥ f x := by
  sorry

end maximize_f_at_1_5_l1645_164540


namespace bing_location_subject_l1645_164502

-- Defining entities
inductive City
| Beijing
| Shanghai
| Chongqing

inductive Subject
| Mathematics
| Chinese
| ForeignLanguage

inductive Teacher
| Jia
| Yi
| Bing

-- Defining the conditions
variables (works_in : Teacher → City) (teaches : Teacher → Subject)

axiom cond1_jia_not_beijing : works_in Teacher.Jia ≠ City.Beijing
axiom cond1_yi_not_shanghai : works_in Teacher.Yi ≠ City.Shanghai
axiom cond2_beijing_not_foreign : ∀ t, works_in t = City.Beijing → teaches t ≠ Subject.ForeignLanguage
axiom cond3_shanghai_math : ∀ t, works_in t = City.Shanghai → teaches t = Subject.Mathematics
axiom cond4_yi_not_chinese : teaches Teacher.Yi ≠ Subject.Chinese

-- The question
theorem bing_location_subject : 
  works_in Teacher.Bing = City.Beijing ∧ teaches Teacher.Bing = Subject.Chinese :=
by
  sorry

end bing_location_subject_l1645_164502


namespace larry_final_channels_l1645_164582

def initial_channels : Int := 150
def removed_channels : Int := 20
def replacement_channels : Int := 12
def reduced_channels : Int := 10
def sports_package_channels : Int := 8
def supreme_sports_package_channels : Int := 7

theorem larry_final_channels :
  initial_channels 
  - removed_channels 
  + replacement_channels 
  - reduced_channels 
  + sports_package_channels 
  + supreme_sports_package_channels 
  = 147 := by
  rfl  -- Reflects the direct computation as per the problem

end larry_final_channels_l1645_164582


namespace shortest_player_height_correct_l1645_164578

def tallest_player_height : Real := 77.75
def height_difference : Real := 9.5
def shortest_player_height : Real := 68.25

theorem shortest_player_height_correct :
  tallest_player_height - height_difference = shortest_player_height :=
by
  sorry

end shortest_player_height_correct_l1645_164578


namespace feet_perpendiculars_concyclic_l1645_164563

variables {S A B C D O M N P Q : Type} 

-- Given conditions
variables (is_convex_quadrilateral : convex_quadrilateral A B C D)
variables (diagonals_perpendicular : ∀ (AC BD : Line), perpendicular AC BD)
variables (foot_perpendicular : ∀ (O : Point), intersection_point O = foot (perpendicular_from S (base_quadrilateral A B C D)))

-- Define the proof statement
theorem feet_perpendiculars_concyclic
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_perpendicular AC BD)
  (h3 : foot_perpendicular O) :
  concyclic (feet_perpendicular_pts O (face S A B)) (feet_perpendicular_pts O (face S B C)) 
            (feet_perpendicular_pts O (face S C D)) (feet_perpendicular_pts O (face S D A)) := sorry

end feet_perpendiculars_concyclic_l1645_164563


namespace burritos_in_each_box_l1645_164587

theorem burritos_in_each_box (B : ℕ) (h1 : 3 * B - B - 30 = 10) : B = 20 :=
by
  sorry

end burritos_in_each_box_l1645_164587


namespace elements_in_M_l1645_164588

def is_element_of_M (x y : ℕ) : Prop :=
  x + y ≤ 1

def M : Set (ℕ × ℕ) :=
  {p | is_element_of_M p.fst p.snd}

theorem elements_in_M :
  M = { (0,0), (0,1), (1,0) } :=
by
  -- Proof would go here
  sorry

end elements_in_M_l1645_164588


namespace A_not_losing_prob_correct_l1645_164521

def probability_draw : ℚ := 1 / 2
def probability_A_wins : ℚ := 1 / 3
def probability_A_not_losing : ℚ := 5 / 6

theorem A_not_losing_prob_correct : 
  probability_draw + probability_A_wins = probability_A_not_losing := 
by sorry

end A_not_losing_prob_correct_l1645_164521


namespace sum_of_cosines_l1645_164552

theorem sum_of_cosines :
  (Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (6 * Real.pi / 7) = -1 / 2) := sorry

end sum_of_cosines_l1645_164552


namespace riding_mower_speed_l1645_164543

theorem riding_mower_speed :
  (∃ R : ℝ, 
     (8 * (3 / 4) = 6) ∧       -- Jerry mows 6 acres with the riding mower
     (8 * (1 / 4) = 2) ∧       -- Jerry mows 2 acres with the push mower
     (2 / 1 = 2) ∧             -- Push mower takes 2 hours to mow 2 acres
     (5 - 2 = 3) ∧             -- Time spent on the riding mower is 3 hours
     (6 / 3 = R) ∧             -- Riding mower cuts 6 acres in 3 hours
     R = 2) :=                 -- Therefore, R (speed of riding mower in acres per hour) is 2
sorry

end riding_mower_speed_l1645_164543


namespace cone_apex_angle_l1645_164504

theorem cone_apex_angle (R : ℝ) 
  (h1 : ∀ (θ : ℝ), (∃ (r : ℝ), r = R / 2 ∧ 2 * π * r = π * R)) :
  ∀ (θ : ℝ), θ = π / 3 :=
by
  sorry

end cone_apex_angle_l1645_164504


namespace equivalent_statements_l1645_164572

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by
  sorry

end equivalent_statements_l1645_164572


namespace dave_used_tickets_for_toys_l1645_164518

-- Define the given conditions
def number_of_tickets_won : ℕ := 18
def tickets_more_for_clothes : ℕ := 10

-- Define the main conjecture
theorem dave_used_tickets_for_toys (T : ℕ) : T + (T + tickets_more_for_clothes) = number_of_tickets_won → T = 4 :=
by {
  -- We'll need the proof here, but it's not required for the statement purpose.
  sorry
}

end dave_used_tickets_for_toys_l1645_164518


namespace cistern_length_l1645_164561

variable (L : ℝ) (width water_depth total_area : ℝ)

theorem cistern_length
  (h_width : width = 8)
  (h_water_depth : water_depth = 1.5)
  (h_total_area : total_area = 134) :
  11 * L + 24 = total_area → L = 10 :=
by
  intro h_eq
  have h_eq1 : 11 * L = 110 := by
    linarith
  have h_L : L = 10 := by
    linarith
  exact h_L

end cistern_length_l1645_164561


namespace prod_ab_eq_three_l1645_164570

theorem prod_ab_eq_three (a b : ℝ) (h₁ : a - b = 5) (h₂ : a^2 + b^2 = 31) : a * b = 3 := 
sorry

end prod_ab_eq_three_l1645_164570


namespace find_a_b_find_max_m_l1645_164575

-- Define the function
def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (3 * x - 2)

-- Conditions
def solution_set_condition (x a : ℝ) : Prop := (-4 * a / 5 ≤ x ∧ x ≤ 3 * a / 5)
def eq_five_condition (x : ℝ) : Prop := f x ≤ 5

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) : (∀ x : ℝ, eq_five_condition x ↔ solution_set_condition x a) → (a = 1 ∧ b = 2) :=
by
  sorry

-- Prove that |x - a| + |x + b| >= m^2 - 3m and find the maximum value of m
theorem find_max_m (a b m : ℝ) : (a = 1 ∧ b = 2) →
  (∀ x : ℝ, abs (x - a) + abs (x + b) ≥ m^2 - 3 * m) →
  m ≤ (3 + Real.sqrt 21) / 2 :=
by
  sorry


end find_a_b_find_max_m_l1645_164575


namespace repair_cost_l1645_164512

theorem repair_cost (purchase_price transport_cost sale_price : ℝ) (profit_percentage : ℝ) (repair_cost : ℝ) :
  purchase_price = 14000 →
  transport_cost = 1000 →
  sale_price = 30000 →
  profit_percentage = 50 →
  sale_price = (1 + profit_percentage / 100) * (purchase_price + repair_cost + transport_cost) →
  repair_cost = 5000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end repair_cost_l1645_164512


namespace infinitely_many_triples_of_integers_l1645_164510

theorem infinitely_many_triples_of_integers (k : ℕ) :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧
                  (x^999 + y^1000 = z^1001) :=
by
  sorry

end infinitely_many_triples_of_integers_l1645_164510


namespace bianca_birthday_money_l1645_164548

/-- Define the number of friends Bianca has -/
def number_of_friends : ℕ := 5

/-- Define the amount of dollars each friend gave -/
def dollars_per_friend : ℕ := 6

/-- The total amount of dollars Bianca received -/
def total_dollars_received : ℕ := number_of_friends * dollars_per_friend

/-- Prove that the total amount of dollars Bianca received is 30 -/
theorem bianca_birthday_money : total_dollars_received = 30 :=
by
  sorry

end bianca_birthday_money_l1645_164548


namespace time_gaps_l1645_164519

theorem time_gaps (dist_a dist_b dist_c : ℕ) (time_a time_b time_c : ℕ) :
  dist_a = 130 →
  dist_b = 130 →
  dist_c = 130 →
  time_a = 36 →
  time_b = 45 →
  time_c = 42 →
  (time_b - time_a = 9) ∧ (time_c - time_a = 6) ∧ (time_b - time_c = 3) := by
  intros hdist_a hdist_b hdist_c htime_a htime_b htime_c
  sorry

end time_gaps_l1645_164519


namespace union_of_sets_l1645_164526

def M : Set Int := { -1, 0, 1 }
def N : Set Int := { 0, 1, 2 }

theorem union_of_sets : M ∪ N = { -1, 0, 1, 2 } := by
  sorry

end union_of_sets_l1645_164526


namespace total_promotional_items_l1645_164541

def num_calendars : ℕ := 300
def num_date_books : ℕ := 200

theorem total_promotional_items : num_calendars + num_date_books = 500 := by
  sorry

end total_promotional_items_l1645_164541


namespace composite_shape_perimeter_l1645_164506

theorem composite_shape_perimeter :
  let r1 := 2.1
  let r2 := 3.6
  let π_approx := 3.14159
  let total_perimeter := π_approx * (r1 + r2)
  total_perimeter = 18.31 :=
by
  let radius1 := 2.1
  let radius2 := 3.6
  let total_radius := radius1 + radius2
  let pi_value := 3.14159
  let perimeter := pi_value * total_radius
  have calculation : perimeter = 18.31 := sorry
  exact calculation

end composite_shape_perimeter_l1645_164506


namespace area_unpainted_region_l1645_164597

theorem area_unpainted_region
  (width_board_1 : ℝ)
  (width_board_2 : ℝ)
  (cross_angle_degrees : ℝ)
  (unpainted_area : ℝ)
  (h1 : width_board_1 = 5)
  (h2 : width_board_2 = 7)
  (h3 : cross_angle_degrees = 45)
  (h4 : unpainted_area = (49 * Real.sqrt 2) / 2) : 
  unpainted_area = (width_board_2 * ((width_board_1 * Real.sqrt 2) / 2)) / 2 :=
sorry

end area_unpainted_region_l1645_164597


namespace binary_preceding_and_following_l1645_164514

theorem binary_preceding_and_following :
  ∀ (n : ℕ), n = 0b1010100 → (Nat.pred n = 0b1010011 ∧ Nat.succ n = 0b1010101) := by
  intros
  sorry

end binary_preceding_and_following_l1645_164514


namespace possible_values_x2_y2_z2_l1645_164576

theorem possible_values_x2_y2_z2 {x y z : ℤ}
    (h1 : x + y + z = 3)
    (h2 : x^3 + y^3 + z^3 = 3) : (x^2 + y^2 + z^2 = 3) ∨ (x^2 + y^2 + z^2 = 57) :=
by sorry

end possible_values_x2_y2_z2_l1645_164576


namespace functional_equation_solution_l1645_164556

/-- For all functions f: ℝ → ℝ, that satisfy the given functional equation -/
def functional_equation (f: ℝ → ℝ) : Prop :=
  ∀ x y: ℝ, f (x + y * f (x + y)) = y ^ 2 + f (x * f (y + 1))

/-- The solution to the functional equation is f(x) = x -/
theorem functional_equation_solution :
  ∀ f: ℝ → ℝ, functional_equation f → (∀ x: ℝ, f x = x) :=
by
  intros f h x
  sorry

end functional_equation_solution_l1645_164556


namespace notebook_pen_cost_l1645_164517

theorem notebook_pen_cost :
  ∃ (n p : ℕ), 15 * n + 4 * p = 160 ∧ n > p ∧ n + p = 18 := 
sorry

end notebook_pen_cost_l1645_164517


namespace evaluate_expression_l1645_164508

theorem evaluate_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 8^(3/2) + 5) = 25 - 16 * Real.sqrt 2 := 
by
  sorry

end evaluate_expression_l1645_164508


namespace cistern_water_depth_l1645_164589

theorem cistern_water_depth
  (length width : ℝ) 
  (wet_surface_area : ℝ)
  (h : ℝ) 
  (hl : length = 7)
  (hw : width = 4)
  (ha : wet_surface_area = 55.5)
  (h_eq : 28 + 22 * h = wet_surface_area) 
  : h = 1.25 := 
  by 
  sorry

end cistern_water_depth_l1645_164589


namespace find_m_for_asymptotes_l1645_164503

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

-- Definition of the asymptotes form
def asymptote_form (m : ℝ) (x y : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

-- The main theorem to prove
theorem find_m_for_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptote_form (4 / 3) x y) :=
sorry

end find_m_for_asymptotes_l1645_164503


namespace square_side_length_l1645_164516

-- Variables for the conditions
variables (totalWire triangleWire : ℕ)
-- Definitions of the conditions
def totalLengthCondition := totalWire = 78
def triangleLengthCondition := triangleWire = 46

-- Goal is to prove the side length of the square
theorem square_side_length
  (h1 : totalLengthCondition totalWire)
  (h2 : triangleLengthCondition triangleWire)
  : (totalWire - triangleWire) / 4 = 8 := 
by
  rw [totalLengthCondition, triangleLengthCondition] at *
  sorry

end square_side_length_l1645_164516


namespace intersection_count_sum_l1645_164538

theorem intersection_count_sum : 
  let m := 252
  let n := 252
  m + n = 504 := 
by {
  let m := 252 
  let n := 252 
  exact Eq.refl 504
}

end intersection_count_sum_l1645_164538


namespace maximum_squares_formation_l1645_164590

theorem maximum_squares_formation (total_matchsticks : ℕ) (triangles : ℕ) (used_for_triangles : ℕ) (remaining_matchsticks : ℕ) (squares : ℕ):
  total_matchsticks = 24 →
  triangles = 6 →
  used_for_triangles = 13 →
  remaining_matchsticks = total_matchsticks - used_for_triangles →
  squares = remaining_matchsticks / 4 →
  squares = 4 :=
by
  sorry

end maximum_squares_formation_l1645_164590


namespace most_noteworthy_figure_is_mode_l1645_164525

-- Define the types of possible statistics
inductive Statistic
| Median
| Mean
| Mode
| WeightedMean

-- Define a structure for survey data (details abstracted)
structure SurveyData where
  -- fields abstracted for this problem

-- Define the concept of the most noteworthy figure
def most_noteworthy_figure (data : SurveyData) : Statistic :=
  Statistic.Mode

-- Theorem to prove the most noteworthy figure in a survey's data is the mode
theorem most_noteworthy_figure_is_mode (data : SurveyData) :
  most_noteworthy_figure data = Statistic.Mode :=
by
  sorry

end most_noteworthy_figure_is_mode_l1645_164525


namespace students_in_class_l1645_164593

theorem students_in_class (x : ℕ) (S : ℕ)
  (h1 : S = 3 * (S / x) + 24)
  (h2 : S = 4 * (S / x) - 26) : 3 * x + 24 = 4 * x - 26 :=
by
  sorry

end students_in_class_l1645_164593


namespace average_marks_class_l1645_164585

theorem average_marks_class (total_students : ℕ)
  (students_98 : ℕ) (score_98 : ℕ)
  (students_0 : ℕ) (score_0 : ℕ)
  (remaining_avg : ℝ)
  (h1 : total_students = 40)
  (h2 : students_98 = 6)
  (h3 : score_98 = 98)
  (h4 : students_0 = 9)
  (h5 : score_0 = 0)
  (h6 : remaining_avg = 57) :
  ( (( students_98 * score_98) + (students_0 * score_0) + ((total_students - students_98 - students_0) * remaining_avg)) / total_students ) = 50.325 :=
by 
  -- This is where the proof steps would go
  sorry

end average_marks_class_l1645_164585


namespace concert_ticket_to_motorcycle_ratio_l1645_164559

theorem concert_ticket_to_motorcycle_ratio (initial_amount spend_motorcycle remaining_amount : ℕ)
  (h_initial : initial_amount = 5000)
  (h_spend_motorcycle : spend_motorcycle = 2800)
  (amount_left := initial_amount - spend_motorcycle)
  (h_remaining : remaining_amount = 825)
  (h_amount_left : ∃ C : ℕ, amount_left - C - (1/4 : ℚ) * (amount_left - C) = remaining_amount) :
  ∃ C : ℕ, (C / amount_left) = (1 / 2 : ℚ) := sorry

end concert_ticket_to_motorcycle_ratio_l1645_164559


namespace common_difference_of_AP_l1645_164532

theorem common_difference_of_AP (a T_12 : ℝ) (d : ℝ) (n : ℕ) (h1 : a = 2) (h2 : T_12 = 90) (h3 : n = 12) 
(h4 : T_12 = a + (n - 1) * d) : d = 8 := 
by sorry

end common_difference_of_AP_l1645_164532


namespace total_distance_swam_l1645_164550

theorem total_distance_swam (molly_swam_saturday : ℕ) (molly_swam_sunday : ℕ) (h1 : molly_swam_saturday = 400) (h2 : molly_swam_sunday = 300) : molly_swam_saturday + molly_swam_sunday = 700 := by 
    sorry

end total_distance_swam_l1645_164550


namespace cos_beta_calculation_l1645_164542

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
variable (h2 : 0 < β ∧ β < π / 2) -- β is an acute angle
variable (h3 : Real.cos α = Real.sqrt 5 / 5)
variable (h4 : Real.sin (α - β) = Real.sqrt 10 / 10)

theorem cos_beta_calculation :
  Real.cos β = Real.sqrt 2 / 2 :=
  sorry

end cos_beta_calculation_l1645_164542


namespace polynomial_at_five_l1645_164594

theorem polynomial_at_five (P : ℝ → ℝ) 
  (hP_degree : ∃ (a b c d : ℝ), ∀ x : ℝ, P x = a*x^3 + b*x^2 + c*x + d)
  (hP1 : P 1 = 1 / 3)
  (hP2 : P 2 = 1 / 7)
  (hP3 : P 3 = 1 / 13)
  (hP4 : P 4 = 1 / 21) :
  P 5 = -3 / 91 :=
sorry

end polynomial_at_five_l1645_164594


namespace find_the_number_l1645_164581

theorem find_the_number (x : ℤ) (h : 2 + x = 6) : x = 4 :=
sorry

end find_the_number_l1645_164581


namespace interest_earned_l1645_164580

theorem interest_earned :
  let P : ℝ := 1500
  let r : ℝ := 0.02
  let n : ℕ := 3
  let A : ℝ := P * (1 + r) ^ n
  let interest : ℝ := A - P
  interest = 92 := 
by
  sorry

end interest_earned_l1645_164580


namespace total_sections_l1645_164596

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls) = 29 :=
by
  sorry

end total_sections_l1645_164596


namespace smallest_invariant_number_l1645_164592

def operation (n : ℕ) : ℕ :=
  let q := n / 10
  let r := n % 10
  q + 2 * r

def is_invariant (n : ℕ) : Prop :=
  operation n = n

theorem smallest_invariant_number : ∃ n : ℕ, is_invariant n ∧ n = 10^99 + 1 :=
by
  sorry

end smallest_invariant_number_l1645_164592


namespace percent_games_lost_l1645_164566

def games_ratio (won lost : ℕ) : Prop :=
  won * 3 = lost * 7

def total_games (won lost : ℕ) : Prop :=
  won + lost = 50

def percentage_lost (lost total : ℕ) : ℕ :=
  lost * 100 / total

theorem percent_games_lost (won lost : ℕ) (h1 : games_ratio won lost) (h2 : total_games won lost) : 
  percentage_lost lost 50 = 30 := 
by
  sorry

end percent_games_lost_l1645_164566


namespace length_segment_FF_l1645_164545

-- Define the points F and F' based on the given conditions
def F : (ℝ × ℝ) := (4, 3)
def F' : (ℝ × ℝ) := (-4, 3)

-- The theorem to prove the length of the segment FF' is 8
theorem length_segment_FF' : dist F F' = 8 :=
by
  sorry

end length_segment_FF_l1645_164545


namespace remainder_when_sum_divided_by_11_l1645_164537

def sum_of_large_numbers : ℕ :=
  100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007

theorem remainder_when_sum_divided_by_11 : sum_of_large_numbers % 11 = 2 := by
  sorry

end remainder_when_sum_divided_by_11_l1645_164537


namespace volume_of_normal_block_is_3_l1645_164549

variable (w d l : ℝ)
def V_normal : ℝ := w * d * l
def V_large : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem volume_of_normal_block_is_3 (h : V_large w d l = 36) : V_normal w d l = 3 :=
by sorry

end volume_of_normal_block_is_3_l1645_164549


namespace expression_evaluation_l1645_164528

noncomputable def x : ℝ := (Real.sqrt 1.21) ^ 3
noncomputable def y : ℝ := (Real.sqrt 0.81) ^ 2
noncomputable def a : ℝ := 4 * Real.sqrt 0.81
noncomputable def b : ℝ := 2 * Real.sqrt 0.49
noncomputable def c : ℝ := 3 * Real.sqrt 1.21
noncomputable def d : ℝ := 2 * Real.sqrt 0.49
noncomputable def e : ℝ := (Real.sqrt 0.81) ^ 4

theorem expression_evaluation : ((x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))) = 1.291343 := by 
  sorry

end expression_evaluation_l1645_164528


namespace part_a_l1645_164565

-- Lean 4 statement equivalent to Part (a)
theorem part_a (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : n^2 ≤ x) : 
  n * Real.sqrt (x - n^2) ≤ x / 2 := 
sorry

-- Lean 4 statement equivalent to Part (b)
noncomputable def find_xyz : ℕ × ℕ × ℕ :=
  ((2, 8, 18) : ℕ × ℕ × ℕ)

end part_a_l1645_164565


namespace compute_expression_l1645_164544

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l1645_164544


namespace harris_carrot_cost_l1645_164557

-- Definitions stemming from the conditions
def carrots_per_day : ℕ := 1
def days_per_year : ℕ := 365
def carrots_per_bag : ℕ := 5
def cost_per_bag : ℕ := 2

-- Prove that Harris's total cost for carrots in one year is $146
theorem harris_carrot_cost : (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag = 146 := by
  sorry

end harris_carrot_cost_l1645_164557


namespace bracket_mul_l1645_164500

def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem bracket_mul : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_mul_l1645_164500


namespace find_age_of_second_person_l1645_164584

variable (T A X : ℝ)

def average_original_group (T A : ℝ) : Prop :=
  T = 7 * A

def average_with_39 (T A : ℝ) : Prop :=
  T + 39 = 8 * (A + 2)

def average_with_second_person (T A X : ℝ) : Prop :=
  T + X = 8 * (A - 1) 

theorem find_age_of_second_person (T A X : ℝ) 
  (h1 : average_original_group T A)
  (h2 : average_with_39 T A)
  (h3 : average_with_second_person T A X) :
  X = 15 :=
sorry

end find_age_of_second_person_l1645_164584


namespace domain_of_g_l1645_164558

theorem domain_of_g (x y : ℝ) : 
  (∃ g : ℝ, g = 1 / (x^2 + (x - y)^2 + y^2)) ↔ (x, y) ≠ (0, 0) :=
by sorry

end domain_of_g_l1645_164558


namespace carol_betty_age_ratio_l1645_164531

theorem carol_betty_age_ratio:
  ∀ (C A B : ℕ), 
    C = 5 * A → 
    A = C - 12 → 
    B = 6 → 
    C / B = 5 / 2 :=
by
  intros C A B h1 h2 h3
  sorry

end carol_betty_age_ratio_l1645_164531


namespace find_m_plus_n_l1645_164539

-- Definitions
structure Triangle (A B C P M N : Type) :=
  (midpoint_AD_P : P)
  (intersection_M_AB : M)
  (intersection_N_AC : N)
  (vec_AB : ℝ)
  (vec_AM : ℝ)
  (vec_AC : ℝ)
  (vec_AN : ℝ)
  (m : ℝ)
  (n : ℝ)
  (AB_eq_AM_mul_m : vec_AB = m * vec_AM)
  (AC_eq_AN_mul_n : vec_AC = n * vec_AN)

-- The theorem to prove
theorem find_m_plus_n (A B C P M N : Type)
  (t : Triangle A B C P M N) :
  t.m + t.n = 4 :=
sorry

end find_m_plus_n_l1645_164539


namespace percentage_of_students_attend_chess_class_l1645_164579

-- Definitions based on the conditions
def total_students : ℕ := 1000
def swimming_students : ℕ := 125
def chess_to_swimming_ratio : ℚ := 1 / 2

-- Problem statement
theorem percentage_of_students_attend_chess_class :
  ∃ P : ℚ, (P / 100) * total_students / 2 = swimming_students → P = 25 := by
  sorry

end percentage_of_students_attend_chess_class_l1645_164579


namespace real_solutions_quadratic_l1645_164568

theorem real_solutions_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 4 * x + a = 0) ↔ a ≤ 4 :=
by sorry

end real_solutions_quadratic_l1645_164568


namespace book_length_l1645_164546

theorem book_length (P : ℕ) (h1 : 2323 = (P - 2323) + 90) : P = 4556 :=
by
  sorry

end book_length_l1645_164546


namespace tagged_fish_ratio_l1645_164574

theorem tagged_fish_ratio (tagged_first_catch : ℕ) (total_second_catch : ℕ) (tagged_second_catch : ℕ) 
  (approx_total_fish : ℕ) (h1 : tagged_first_catch = 60) 
  (h2 : total_second_catch = 50) 
  (h3 : tagged_second_catch = 2) 
  (h4 : approx_total_fish = 1500) :
  tagged_second_catch / total_second_catch = 1 / 25 := by
  sorry

end tagged_fish_ratio_l1645_164574


namespace find_xy_pairs_l1645_164522

theorem find_xy_pairs (x y: ℝ) :
  x + y + 4 = (12 * x + 11 * y) / (x ^ 2 + y ^ 2) ∧
  y - x + 3 = (11 * x - 12 * y) / (x ^ 2 + y ^ 2) ↔
  (x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5) :=
by
  sorry

end find_xy_pairs_l1645_164522


namespace bus_passengers_final_count_l1645_164501

theorem bus_passengers_final_count :
  let initial_passengers := 15
  let changes := [(3, -6), (-2, 4), (-7, 2), (3, -5)]
  let apply_change (acc : Int) (change : Int × Int) : Int :=
    acc + change.1 + change.2
  initial_passengers + changes.foldl apply_change 0 = 7 :=
by
  intros
  sorry

end bus_passengers_final_count_l1645_164501


namespace sunil_investment_l1645_164591

noncomputable def total_amount (P : ℝ) : ℝ :=
  let r1 := 0.025  -- 5% per annum compounded semi-annually
  let r2 := 0.03   -- 6% per annum compounded semi-annually
  let A2 := P * (1 + r1) * (1 + r1)
  let A3 := (A2 + 0.5 * P) * (1 + r2)
  let A4 := A3 * (1 + r2)
  A4

theorem sunil_investment (P : ℝ) : total_amount P = 1.645187625 * P :=
by
  sorry

end sunil_investment_l1645_164591


namespace AM_GM_inequality_l1645_164569

theorem AM_GM_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2) ^ n :=
by
  sorry

end AM_GM_inequality_l1645_164569


namespace bus_time_l1645_164553

variable (t1 t2 t3 t4 : ℕ)

theorem bus_time
  (h1 : t1 = 25)
  (h2 : t2 = 40)
  (h3 : t3 = 15)
  (h4 : t4 = 10) :
  t1 + t2 + t3 + t4 = 90 := by
  sorry

end bus_time_l1645_164553


namespace father_son_age_relationship_l1645_164560

theorem father_son_age_relationship 
    (F S X : ℕ) 
    (h1 : F = 27) 
    (h2 : F = 3 * S + 3) 
    : X = 11 ∧ F + X > 2 * (S + X) :=
by
  sorry

end father_son_age_relationship_l1645_164560


namespace survey_representative_l1645_164529

universe u

inductive SurveyOption : Type u
| A : SurveyOption  -- Selecting a class of students
| B : SurveyOption  -- Selecting 50 male students
| C : SurveyOption  -- Selecting 50 female students
| D : SurveyOption  -- Randomly selecting 50 eighth-grade students

def most_appropriate_survey : SurveyOption := SurveyOption.D

theorem survey_representative : most_appropriate_survey = SurveyOption.D := 
by sorry

end survey_representative_l1645_164529


namespace cost_of_hard_lenses_l1645_164527

theorem cost_of_hard_lenses (x H : ℕ) (h1 : x + (x + 5) = 11)
    (h2 : 150 * (x + 5) + H * x = 1455) : H = 85 := by
  sorry

end cost_of_hard_lenses_l1645_164527


namespace algebra_expression_value_l1645_164536

theorem algebra_expression_value (a : ℝ) (h : 3 * a ^ 2 + 2 * a - 1 = 0) : 3 * a ^ 2 + 2 * a - 2019 = -2018 := 
by 
  -- Proof goes here
  sorry

end algebra_expression_value_l1645_164536


namespace circle_eq_problem1_circle_eq_problem2_l1645_164509

-- Problem 1
theorem circle_eq_problem1 :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2 ∧
  a - 2 * b - 3 = 0 ∧
  (2 - a)^2 + (-3 - b)^2 = r^2 ∧
  (-2 - a)^2 + (-5 - b)^2 = r^2) ↔
  (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Problem 2
theorem circle_eq_problem2 :
  (∃ D E F : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ∧
  (1:ℝ)^2 + (0:ℝ)^2 + D * 1 + E * 0 + F = 0 ∧
  (-1:ℝ)^2 + (-2:ℝ)^2 - D * 1 - 2 * E + F = 0 ∧
  (3:ℝ)^2 + (-2:ℝ)^2 + 3 * D - 2 * E + F = 0) ↔
  x^2 + y^2 - 2 * x + 4 * y + 1 = 0 :=
sorry

end circle_eq_problem1_circle_eq_problem2_l1645_164509


namespace problem_statement_l1645_164573

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 : Prop := a^2 + b^2 - 4 * a ≤ 1
def condition2 : Prop := b^2 + c^2 - 8 * b ≤ -3
def condition3 : Prop := c^2 + a^2 - 12 * c ≤ -26

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c a) : (a + b) ^ c = 27 :=
by sorry

end problem_statement_l1645_164573


namespace find_x_plus_y_l1645_164567

variables {x y : ℝ}

def f (t : ℝ) : ℝ := t^2003 + 2002 * t

theorem find_x_plus_y (hx : f (x - 1) = -1) (hy : f (y - 2) = 1) : x + y = 3 :=
by
  sorry

end find_x_plus_y_l1645_164567


namespace gary_has_left_amount_l1645_164555

def initial_amount : ℝ := 100
def cost_pet_snake : ℝ := 55
def cost_toy_car : ℝ := 12
def cost_novel : ℝ := 7.5
def cost_pack_stickers : ℝ := 3.25
def number_packs_stickers : ℕ := 3

theorem gary_has_left_amount : initial_amount - (cost_pet_snake + cost_toy_car + cost_novel + number_packs_stickers * cost_pack_stickers) = 15.75 :=
by
  sorry

end gary_has_left_amount_l1645_164555


namespace square_area_l1645_164511

theorem square_area (s : ℕ) (h : s = 13) : s * s = 169 := by
  sorry

end square_area_l1645_164511


namespace greatest_integer_of_negative_fraction_l1645_164583

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end greatest_integer_of_negative_fraction_l1645_164583


namespace max_elevation_l1645_164586

def particle_elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

theorem max_elevation : ∃ t : ℝ, particle_elevation t = 550 :=
by {
  sorry
}

end max_elevation_l1645_164586


namespace solve_inequality_l1645_164534

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∨ x ∈ Set.Icc 2 4) :=
sorry

end solve_inequality_l1645_164534


namespace people_in_room_after_2019_minutes_l1645_164595

theorem people_in_room_after_2019_minutes :
  ∀ (P : Nat → Int), 
    P 0 = 0 -> 
    (∀ t, P (t+1) = P t + 2 ∨ P (t+1) = P t - 1) -> 
    P 2019 ≠ 2018 :=
by
  intros P hP0 hP_changes
  sorry

end people_in_room_after_2019_minutes_l1645_164595
