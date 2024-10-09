import Mathlib

namespace cosine_of_five_pi_over_three_l908_90894

theorem cosine_of_five_pi_over_three :
  Real.cos (5 * Real.pi / 3) = 1 / 2 :=
sorry

end cosine_of_five_pi_over_three_l908_90894


namespace increase_in_average_commission_l908_90837

theorem increase_in_average_commission :
  ∀ (new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 : ℕ),
    new_avg = 400 → 
    n1 = 6 → 
    n2 = n1 - 1 → 
    big_sale = 1300 →
    total_earnings = new_avg * n1 →
    commission = total_earnings - big_sale →
    old_avg = commission / n2 →
    new_avg - old_avg = 180 :=
by 
  intros new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end increase_in_average_commission_l908_90837


namespace rational_function_nonnegative_l908_90826

noncomputable def rational_function (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3)

theorem rational_function_nonnegative :
  ∀ x, 0 ≤ x ∧ x < 3 → 0 ≤ rational_function x :=
sorry

end rational_function_nonnegative_l908_90826


namespace simple_interest_years_l908_90898

noncomputable def simple_interest (P r t : ℕ) : ℕ :=
  P * r * t / 100

noncomputable def compound_interest (P r n : ℕ) : ℕ :=
  P * (1 + r / 100)^n - P

theorem simple_interest_years
  (P_si r_si P_ci r_ci n_ci si_half_ci si_si : ℕ)
  (h_si : simple_interest P_si r_si si_si = si_half_ci)
  (h_ci : compound_interest P_ci r_ci n_ci = si_half_ci * 2) :
  si_si = 2 :=
by
  sorry

end simple_interest_years_l908_90898


namespace correct_expression_l908_90892

-- Definitions based on given conditions
def expr1 (a b : ℝ) := 3 * a + 2 * b = 5 * a * b
def expr2 (a : ℝ) := 2 * a^3 - a^3 = a^3
def expr3 (a b : ℝ) := a^2 * b - a * b = a
def expr4 (a : ℝ) := a^2 + a^2 = 2 * a^4

-- Statement to prove that expr2 is the only correct expression
theorem correct_expression (a b : ℝ) : 
  expr2 a := by
  sorry

end correct_expression_l908_90892


namespace solve_equation_l908_90864

theorem solve_equation (x y : ℕ) (h_xy : x ≠ y) : x = 2 ∧ y = 4 ∨ x = 4 ∧ y = 2 :=
by {
  sorry -- Proof skipped
}

end solve_equation_l908_90864


namespace at_least_two_consecutive_heads_probability_l908_90818

theorem at_least_two_consecutive_heads_probability :
  let outcomes := ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
  let favorable_outcomes := ["HHH", "HHT", "THH"]
  let total_outcomes := outcomes.length
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 2 :=
by sorry

end at_least_two_consecutive_heads_probability_l908_90818


namespace compute_f_g_2_l908_90861

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem compute_f_g_2 : f (g 2) = -19 := 
by {
  sorry
}

end compute_f_g_2_l908_90861


namespace shopkeeper_standard_weight_l908_90882

theorem shopkeeper_standard_weight
    (cost_price : ℝ)
    (actual_weight_used : ℝ)
    (profit_percentage : ℝ)
    (standard_weight : ℝ)
    (H1 : actual_weight_used = 800)
    (H2 : profit_percentage = 25) :
    standard_weight = 1000 :=
by 
    sorry

end shopkeeper_standard_weight_l908_90882


namespace ratio_of_screams_to_hours_l908_90897

-- Definitions from conditions
def hours_hired : ℕ := 6
def current_babysitter_rate : ℕ := 16
def new_babysitter_rate : ℕ := 12
def extra_charge_per_scream : ℕ := 3
def cost_difference : ℕ := 18

-- Calculate necessary costs
def current_babysitter_cost : ℕ := current_babysitter_rate * hours_hired
def new_babysitter_base_cost : ℕ := new_babysitter_rate * hours_hired
def new_babysitter_total_cost : ℕ := current_babysitter_cost - cost_difference
def screams_cost : ℕ := new_babysitter_total_cost - new_babysitter_base_cost
def number_of_screams : ℕ := screams_cost / extra_charge_per_scream

-- Theorem to prove the ratio
theorem ratio_of_screams_to_hours : number_of_screams / hours_hired = 1 := by
  sorry

end ratio_of_screams_to_hours_l908_90897


namespace compute_expression_l908_90811

theorem compute_expression (x : ℝ) (h : x = 8) : 
  (x^6 - 64 * x^3 + 1024) / (x^3 - 16) = 480 :=
by
  rw [h]
  sorry

end compute_expression_l908_90811


namespace geom_seq_product_arith_seq_l908_90850

theorem geom_seq_product_arith_seq (a b c r : ℝ) (h1 : c = b * r)
  (h2 : b = a * r)
  (h3 : a * b * c = 512)
  (h4 : b = 8)
  (h5 : 2 * b = (a - 2) + (c - 2)) :
  (a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4) :=
by
  sorry

end geom_seq_product_arith_seq_l908_90850


namespace quadratic_no_real_roots_l908_90805

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬(x^2 - 2 * x + 3 = 0) :=
by
  sorry

end quadratic_no_real_roots_l908_90805


namespace number_of_children_tickets_l908_90802

theorem number_of_children_tickets 
    (x y : ℤ) 
    (h1 : x + y = 225) 
    (h2 : 6 * x + 9 * y = 1875) : 
    x = 50 := 
  sorry

end number_of_children_tickets_l908_90802


namespace find_f_zero_function_decreasing_find_range_x_l908_90800

noncomputable def f : ℝ → ℝ := sorry

-- Define the main conditions as hypotheses
axiom additivity : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2
axiom negativity : ∀ x : ℝ, x > 0 → f x < 0

-- First theorem: proving f(0) = 0
theorem find_f_zero : f 0 = 0 := sorry

-- Second theorem: proving the function is decreasing over (-∞, ∞)
theorem function_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

-- Third theorem: finding the range of x such that f(x) + f(2-3x) < 0
theorem find_range_x (x : ℝ) : f x + f (2 - 3 * x) < 0 → x < 1 := sorry

end find_f_zero_function_decreasing_find_range_x_l908_90800


namespace number_of_ways_to_select_book_l908_90825

-- Definitions directly from the problem's conditions
def numMathBooks : Nat := 3
def numChineseBooks : Nat := 5
def numEnglishBooks : Nat := 8

-- The proof problem statement in Lean 4
theorem number_of_ways_to_select_book : numMathBooks + numChineseBooks + numEnglishBooks = 16 := 
by
  show 3 + 5 + 8 = 16
  sorry

end number_of_ways_to_select_book_l908_90825


namespace difference_between_x_and_y_is_36_l908_90817

theorem difference_between_x_and_y_is_36 (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := 
by 
  sorry

end difference_between_x_and_y_is_36_l908_90817


namespace units_digit_2019_pow_2019_l908_90849

theorem units_digit_2019_pow_2019 : (2019^2019) % 10 = 9 := 
by {
  -- The statement of the problem is proved below
  sorry  -- Solution to be filled in
}

end units_digit_2019_pow_2019_l908_90849


namespace train_and_car_combined_time_l908_90877

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l908_90877


namespace factor_polynomial_l908_90859

theorem factor_polynomial :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 = 
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := 
by sorry

end factor_polynomial_l908_90859


namespace peter_has_read_more_books_l908_90801

theorem peter_has_read_more_books
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (brother_percentage : ℚ)
  (sarah_percentage : ℚ)
  (peter_books : ℚ := (peter_percentage / 100) * total_books)
  (brother_books : ℚ := (brother_percentage / 100) * total_books)
  (sarah_books : ℚ := (sarah_percentage / 100) * total_books)
  (combined_books : ℚ := brother_books + sarah_books)
  (difference : ℚ := peter_books - combined_books) :
  total_books = 50 → peter_percentage = 60 → brother_percentage = 25 → sarah_percentage = 15 → difference = 10 :=
by
  sorry

end peter_has_read_more_books_l908_90801


namespace range_of_x_l908_90840

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) (x : ℝ) : 
  (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by 
  sorry

end range_of_x_l908_90840


namespace cheating_percentage_l908_90847

theorem cheating_percentage (x : ℝ) :
  (∀ cost_price : ℝ, cost_price = 100 →
   let received_when_buying : ℝ := cost_price * (1 + x / 100)
   let given_when_selling : ℝ := cost_price * (1 - x / 100)
   let profit : ℝ := received_when_buying - given_when_selling
   let profit_percentage : ℝ := profit / cost_price
   profit_percentage = 2 / 9) →
  x = 22.22222222222222 := 
by
  sorry

end cheating_percentage_l908_90847


namespace num_tents_needed_l908_90899

def count_people : ℕ :=
  let matts_family := 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2
  let joes_family := 1 + 1 + 3 + 1
  matts_family + joes_family

def house_capacity : ℕ := 6

def tent_capacity : ℕ := 2

theorem num_tents_needed : (count_people - house_capacity) / tent_capacity = 7 := by
  sorry

end num_tents_needed_l908_90899


namespace coordinates_of_point_M_l908_90871

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end coordinates_of_point_M_l908_90871


namespace percentage_dogs_and_video_games_l908_90815

theorem percentage_dogs_and_video_games (total_students : ℕ)
  (students_dogs_movies : ℕ)
  (students_prefer_dogs : ℕ) :
  total_students = 30 →
  students_dogs_movies = 3 →
  students_prefer_dogs = 18 →
  (students_prefer_dogs - students_dogs_movies) * 100 / total_students = 50 :=
by
  intros h1 h2 h3
  sorry

end percentage_dogs_and_video_games_l908_90815


namespace candy_distribution_l908_90865

theorem candy_distribution (candy : ℕ) (people : ℕ) (hcandy : candy = 30) (hpeople : people = 5) :
  ∃ k : ℕ, candy - k = people * (candy / people) ∧ k = 0 := 
by
  sorry

end candy_distribution_l908_90865


namespace cos_inequality_l908_90836

open Real

-- Given angles of a triangle A, B, C

theorem cos_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hTriangle : A + B + C = π) :
  1 / (1 + cos B ^ 2 + cos C ^ 2) + 1 / (1 + cos C ^ 2 + cos A ^ 2) + 1 / (1 + cos A ^ 2 + cos B ^ 2) ≤ 2 :=
by
  sorry

end cos_inequality_l908_90836


namespace surface_area_of_sphere_l908_90890

/-- Given a right prism with all vertices on a sphere, a height of 4, and a volume of 64,
    the surface area of this sphere is 48π -/
theorem surface_area_of_sphere (h : ℝ) (V : ℝ) (S : ℝ) :
  h = 4 → V = 64 → S = 48 * Real.pi := by
  sorry

end surface_area_of_sphere_l908_90890


namespace solve_system_l908_90895

theorem solve_system :
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (14.996, 19.994)) ∨
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (0.421, 1.561)) :=
  sorry

end solve_system_l908_90895


namespace projection_of_a_onto_b_l908_90876

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

theorem projection_of_a_onto_b : projection vec_a vec_b = Real.sqrt 5 :=
by
  sorry

end projection_of_a_onto_b_l908_90876


namespace max_value_of_sin_l908_90866

theorem max_value_of_sin (x : ℝ) : (2 * Real.sin x) ≤ 2 :=
by
  -- this theorem directly implies that 2sin(x) has a maximum value of 2.
  sorry

end max_value_of_sin_l908_90866


namespace inequality_proof_l908_90863

variable {x1 x2 y1 y2 z1 z2 : ℝ}

theorem inequality_proof (hx1 : x1 > 0) (hx2 : x2 > 0)
   (hxy1 : x1 * y1 - z1^2 > 0) (hxy2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
  sorry

end inequality_proof_l908_90863


namespace roots_lost_extraneous_roots_l908_90884

noncomputable def f1 (x : ℝ) := Real.arcsin x
noncomputable def g1 (x : ℝ) := 2 * Real.arcsin (x / Real.sqrt 2)
noncomputable def f2 (x : ℝ) := x
noncomputable def g2 (x : ℝ) := 2 * x

theorem roots_lost :
  ∃ x : ℝ, f1 x = g1 x ∧ ¬ ∃ y : ℝ, Real.tan (f1 y) = Real.tan (g1 y) :=
sorry

theorem extraneous_roots :
  ∃ x : ℝ, ¬ f2 x = g2 x ∧ ∃ y : ℝ, Real.tan (f2 y) = Real.tan (g2 y) :=
sorry

end roots_lost_extraneous_roots_l908_90884


namespace deers_distribution_l908_90834

theorem deers_distribution (a_1 d a_2 a_5 : ℚ) 
  (h1 : a_2 = a_1 + d)
  (h2 : 5 * a_1 + 10 * d = 5)
  (h3 : a_2 = 2 / 3) :
  a_5 = 1 / 3 :=
sorry

end deers_distribution_l908_90834


namespace other_root_l908_90855

theorem other_root (k : ℝ) : 
  5 * (2:ℝ)^2 + k * (2:ℝ) - 8 = 0 → 
  ∃ q : ℝ, 5 * q^2 + k * q - 8 = 0 ∧ q ≠ 2 ∧ q = -4/5 :=
by {
  sorry
}

end other_root_l908_90855


namespace P_plus_Q_is_expected_l908_90841

-- defining the set P
def P : Set ℝ := { x | x ^ 2 - 3 * x - 4 ≤ 0 }

-- defining the set Q
def Q : Set ℝ := { x | x ^ 2 - 2 * x - 15 > 0 }

-- defining the set P + Q
def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ ¬(x ∈ P ∧ x ∈ Q) }

-- the expected result
def expected_P_plus_Q : Set ℝ := { x | x < -3 } ∪ { x | -1 ≤ x ∧ x ≤ 4 } ∪ { x | x > 5 }

-- theorem stating that P + Q equals the expected result
theorem P_plus_Q_is_expected : P_plus_Q = expected_P_plus_Q := by
  sorry

end P_plus_Q_is_expected_l908_90841


namespace minutes_after_2017_is_0554_l908_90868

theorem minutes_after_2017_is_0554 :
  let initial_time := (20, 17) -- time in hours and minutes
  let total_minutes := 2017
  let hours_passed := total_minutes / 60
  let minutes_passed := total_minutes % 60
  let days_passed := hours_passed / 24
  let remaining_hours := hours_passed % 24
  let resulting_hours := (initial_time.fst + remaining_hours) % 24
  let resulting_minutes := initial_time.snd + minutes_passed
  let final_hours := if resulting_minutes >= 60 then resulting_hours + 1 else resulting_hours
  let final_minutes := if resulting_minutes >= 60 then resulting_minutes - 60 else resulting_minutes
  final_hours % 24 = 5 ∧ final_minutes = 54 := by
  sorry

end minutes_after_2017_is_0554_l908_90868


namespace total_rats_l908_90851

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l908_90851


namespace find_least_number_subtracted_l908_90873

theorem find_least_number_subtracted (n m : ℕ) (h : n = 78721) (h1 : m = 23) : (n % m) = 15 := by
  sorry

end find_least_number_subtracted_l908_90873


namespace time_to_reach_rest_area_l908_90856

variable (rate_per_minute : ℕ) (remaining_distance_yards : ℕ)

theorem time_to_reach_rest_area (h_rate : rate_per_minute = 2) (h_distance : remaining_distance_yards = 50) :
  (remaining_distance_yards * 3) / rate_per_minute = 75 := by
  sorry

end time_to_reach_rest_area_l908_90856


namespace intersection_point_l908_90824

structure Point3D : Type where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨8, -9, 5⟩
def B : Point3D := ⟨18, -19, 15⟩
def C : Point3D := ⟨2, 5, -8⟩
def D : Point3D := ⟨4, -3, 12⟩

/-- Prove that the intersection point of lines AB and CD is (16, -19, 13) -/
theorem intersection_point :
  ∃ (P : Point3D), 
  (∃ t : ℝ, P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  (∃ s : ℝ, P = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y), C.z + s * (D.z - C.z)⟩) ∧
  P = ⟨16, -19, 13⟩ :=
by
  sorry

end intersection_point_l908_90824


namespace overall_percent_supporters_l908_90819

theorem overall_percent_supporters
  (percent_A : ℝ) (percent_B : ℝ)
  (members_A : ℕ) (members_B : ℕ)
  (supporters_A : ℕ)
  (supporters_B : ℕ)
  (total_supporters : ℕ)
  (total_members : ℕ)
  (overall_percent : ℝ) 
  (h1 : percent_A = 0.70) 
  (h2 : percent_B = 0.75)
  (h3 : members_A = 200) 
  (h4 : members_B = 800) 
  (h5 : supporters_A = percent_A * members_A) 
  (h6 : supporters_B = percent_B * members_B) 
  (h7 : total_supporters = supporters_A + supporters_B) 
  (h8 : total_members = members_A + members_B) 
  (h9 : overall_percent = (total_supporters : ℝ) / total_members * 100) :
  overall_percent = 74 := by
  sorry

end overall_percent_supporters_l908_90819


namespace second_newly_inserted_number_eq_l908_90831

theorem second_newly_inserted_number_eq : 
  ∃ q : ℝ, (q ^ 12 = 2) ∧ (1 * (q ^ 2) = 2 ^ (1 / 6)) := 
by
  sorry

end second_newly_inserted_number_eq_l908_90831


namespace hamburger_cost_l908_90862

def annie's_starting_money : ℕ := 120
def num_hamburgers_bought : ℕ := 8
def price_milkshake : ℕ := 3
def num_milkshakes_bought : ℕ := 6
def leftover_money : ℕ := 70

theorem hamburger_cost :
  ∃ (H : ℕ), 8 * H + 6 * price_milkshake = annie's_starting_money - leftover_money ∧ H = 4 :=
by
  use 4
  sorry

end hamburger_cost_l908_90862


namespace polygon_number_of_sides_and_interior_sum_l908_90844

-- Given conditions
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)
def exterior_angle_sum : ℝ := 360

-- Proof problem statement
theorem polygon_number_of_sides_and_interior_sum (n : ℕ)
  (h : interior_angle_sum n = 3 * exterior_angle_sum) :
  n = 8 ∧ interior_angle_sum n = 1080 :=
by
  sorry

end polygon_number_of_sides_and_interior_sum_l908_90844


namespace inverse_variation_l908_90832

theorem inverse_variation (x y : ℝ) (h1 : 7 * y = 1400 / x^3) (h2 : x = 4) : y = 25 / 8 :=
  by
  sorry

end inverse_variation_l908_90832


namespace volume_of_box_with_ratio_125_l908_90852

def volumes : Finset ℕ := {60, 80, 100, 120, 200}

theorem volume_of_box_with_ratio_125 : 80 ∈ volumes ∧ ∃ (x : ℕ), 10 * x^3 = 80 :=
by {
  -- Skipping the proof, as only the statement is required.
  sorry
}

end volume_of_box_with_ratio_125_l908_90852


namespace part1_part2_part3_l908_90887

-- Part 1
theorem part1 (B_count : ℕ) : 
  (1 * 100) + (B_count * 68) + (4 * 20) = 520 → 
  B_count = 5 := 
by sorry

-- Part 2
theorem part2 (A_count B_count : ℕ) : 
  A_count + B_count = 5 → 
  (100 * A_count) + (68 * B_count) = 404 → 
  A_count = 2 ∧ B_count = 3 := 
by sorry

-- Part 3
theorem part3 : 
  ∃ (A_count B_count C_count : ℕ), 
  (A_count <= 16) ∧ (B_count <= 16) ∧ (C_count <= 16) ∧ 
  (A_count + B_count + C_count <= 16) ∧ 
  (100 * A_count + 68 * B_count = 708 ∨ 
   68 * B_count + 20 * C_count = 708 ∨ 
   100 * A_count + 20 * C_count = 708) → 
  ((A_count = 3 ∧ B_count = 6 ∧ C_count = 0) ∨ 
   (A_count = 0 ∧ B_count = 6 ∧ C_count = 15)) := 
by sorry

end part1_part2_part3_l908_90887


namespace budget_spent_on_research_and_development_l908_90867

theorem budget_spent_on_research_and_development:
  (∀ budget_total : ℝ, budget_total > 0) →
  (∀ transportation : ℝ, transportation = 15) →
  (∃ research_and_development : ℝ, research_and_development ≥ 0) →
  (∀ utilities : ℝ, utilities = 5) →
  (∀ equipment : ℝ, equipment = 4) →
  (∀ supplies : ℝ, supplies = 2) →
  (∀ salaries_degrees : ℝ, salaries_degrees = 234) →
  (∀ total_degrees : ℝ, total_degrees = 360) →
  (∀ percentage_salaries : ℝ, percentage_salaries = (salaries_degrees / total_degrees) * 100) →
  (∀ known_percentages : ℝ, known_percentages = transportation + utilities + equipment + supplies + percentage_salaries) →
  (∀ rnd_percent : ℝ, rnd_percent = 100 - known_percentages) →
  (rnd_percent = 9) :=
  sorry

end budget_spent_on_research_and_development_l908_90867


namespace calculate_exponent_l908_90806

theorem calculate_exponent (m : ℝ) : (243 : ℝ)^(1 / 3) = 3^m → m = 5 / 3 :=
by
  sorry

end calculate_exponent_l908_90806


namespace spencer_sessions_per_day_l908_90846

theorem spencer_sessions_per_day :
  let jumps_per_minute := 4
  let minutes_per_session := 10
  let jumps_per_session := jumps_per_minute * minutes_per_session
  let total_jumps := 400
  let days := 5
  let jumps_per_day := total_jumps / days
  let sessions_per_day := jumps_per_day / jumps_per_session
  sessions_per_day = 2 :=
by
  sorry

end spencer_sessions_per_day_l908_90846


namespace probability_fourth_ball_black_l908_90845

theorem probability_fourth_ball_black :
  let total_balls := 6
  let red_balls := 3
  let black_balls := 3
  let prob_black_first_draw := black_balls / total_balls
  (prob_black_first_draw = 1 / 2) ->
  (prob_black_first_draw = (black_balls / total_balls)) ->
  (black_balls / total_balls = 1 / 2) ->
  1 / 2 = 1 / 2 :=
by
  intros
  sorry

end probability_fourth_ball_black_l908_90845


namespace age_problem_l908_90886

theorem age_problem :
  (∃ (x y : ℕ), 
    (3 * x - 7 = 5 * (x - 7)) ∧ 
    (42 + y = 2 * (14 + y)) ∧ 
    (2 * x = 28) ∧ 
    (x = 14) ∧ 
    (3 * 14 = 42) ∧ 
    (42 - 14 = 28) ∧ 
    (y = 14)) :=
by
  sorry

end age_problem_l908_90886


namespace smallest_positive_z_l908_90821

open Real

theorem smallest_positive_z (x y z : ℝ) (m k n : ℤ) 
  (h1 : cos x = 0) 
  (h2 : sin y = 1) 
  (h3 : cos (x + z) = -1 / 2) :
  z = 5 * π / 6 :=
by
  sorry

end smallest_positive_z_l908_90821


namespace gcd_of_two_powers_l908_90809

-- Define the expressions
def two_pow_1015_minus_1 : ℤ := 2^1015 - 1
def two_pow_1024_minus_1 : ℤ := 2^1024 - 1

-- Define the gcd function and the target value
noncomputable def gcd_expr : ℤ := Int.gcd (2^1015 - 1) (2^1024 - 1)
def target : ℤ := 511

-- The statement we want to prove
theorem gcd_of_two_powers : gcd_expr = target := by 
  sorry

end gcd_of_two_powers_l908_90809


namespace weight_of_balls_l908_90872

theorem weight_of_balls (x y : ℕ) (h1 : 5 * x + 3 * y = 42) (h2 : 5 * y + 3 * x = 38) :
  x = 6 ∧ y = 4 :=
by
  sorry

end weight_of_balls_l908_90872


namespace number_division_l908_90843

theorem number_division (x : ℚ) (h : x / 2 = 100 + x / 5) : x = 1000 / 3 := 
by
  sorry

end number_division_l908_90843


namespace range_of_a_l908_90885

theorem range_of_a 
  (a : ℕ) 
  (an : ℕ → ℕ)
  (Sn : ℕ → ℕ)
  (h1 : a_1 = a)
  (h2 : ∀ n : ℕ, n ≥ 2 → Sn n + Sn (n - 1) = 4 * n^2)
  (h3 : ∀ n : ℕ, an n < an (n + 1)) : 
  3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l908_90885


namespace acute_triangle_inequality_l908_90814

variable (f : ℝ → ℝ)
variable {A B : ℝ}
variable (h₁ : ∀ x : ℝ, x * (f'' x) - 2 * (f x) > 0)
variable (h₂ : A + B < Real.pi / 2 ∧ 0 < A ∧ 0 < B)

theorem acute_triangle_inequality :
  f (Real.cos A) * (Real.sin B) ^ 2 < f (Real.sin B) * (Real.cos A) ^ 2 := 
  sorry

end acute_triangle_inequality_l908_90814


namespace initial_percentage_water_l908_90879

theorem initial_percentage_water (W_initial W_final N_initial N_final : ℝ) (h1 : W_initial = 100) 
    (h2 : N_initial = W_initial - W_final) (h3 : W_final = 25) (h4 : W_final / N_final = 0.96) : N_initial / W_initial = 0.99 := 
by
  sorry

end initial_percentage_water_l908_90879


namespace square_prism_surface_area_eq_volume_l908_90883

theorem square_prism_surface_area_eq_volume :
  ∃ (a b : ℕ), (a > 0) ∧ (2 * a^2 + 4 * a * b = a^2 * b)
  ↔ (a = 12 ∧ b = 3) ∨ (a = 8 ∧ b = 4) ∨ (a = 6 ∧ b = 6) ∨ (a = 5 ∧ b = 10) :=
by
  sorry

end square_prism_surface_area_eq_volume_l908_90883


namespace reciprocal_real_roots_l908_90891

theorem reciprocal_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 * x2 = 1 ∧ x1 + x2 = 2 * (m + 2)) ∧ 
  (x1^2 - 2 * (m + 2) * x1 + (m^2 - 4) = 0) → m = Real.sqrt 5 := 
sorry

end reciprocal_real_roots_l908_90891


namespace constant_function_of_inequality_l908_90881

theorem constant_function_of_inequality
  (f : ℤ → ℝ)
  (h_bound : ∃ M : ℝ, ∀ n : ℤ, f n ≤ M)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n := by
  sorry

end constant_function_of_inequality_l908_90881


namespace simplify_and_evaluate_l908_90880

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  ( (x + 3) / x - 1 ) / ( (x^2 - 1) / (x^2 + x) ) = Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l908_90880


namespace sandy_hourly_wage_l908_90804

theorem sandy_hourly_wage (x : ℝ)
    (h1 : 10 * x + 6 * x + 14 * x = 450) : x = 15 :=
by
    sorry

end sandy_hourly_wage_l908_90804


namespace trapezoid_possible_and_area_sum_l908_90896

theorem trapezoid_possible_and_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 12) :
  ∃ (S : ℚ), S = 72 := 
by
  -- conditions ensure one pair of sides is parallel
  -- area calculation based on trapezoid properties
  sorry

end trapezoid_possible_and_area_sum_l908_90896


namespace r_earns_per_day_l908_90823

variables (P Q R S : ℝ)

theorem r_earns_per_day
  (h1 : P + Q + R + S = 240)
  (h2 : P + R + S = 160)
  (h3 : Q + R = 150)
  (h4 : Q + R + S = 650 / 3) :
  R = 70 :=
by
  sorry

end r_earns_per_day_l908_90823


namespace scientific_notation_l908_90893

theorem scientific_notation : 350000000 = 3.5 * 10^8 :=
by
  sorry

end scientific_notation_l908_90893


namespace day_of_50th_in_year_N_minus_1_l908_90888

theorem day_of_50th_in_year_N_minus_1
  (N : ℕ)
  (day250_in_year_N_is_sunday : (250 % 7 = 0))
  (day150_in_year_N_plus_1_is_sunday : (150 % 7 = 0))
  : 
  (50 % 7 = 1) := 
sorry

end day_of_50th_in_year_N_minus_1_l908_90888


namespace train_length_correct_l908_90848

def train_length (speed_kph : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  speed_mps * time_sec

theorem train_length_correct :
  train_length 90 10 = 250 := by
  sorry

end train_length_correct_l908_90848


namespace sue_shoes_probability_l908_90839

def sueShoes : List (String × ℕ) := [("black", 7), ("brown", 3), ("gray", 2)]

def total_shoes := 24

def prob_same_color (color : String) (pairs : List (String × ℕ)) : ℚ :=
  let total_pairs := pairs.foldr (λ p acc => acc + p.snd) 0
  let matching_pair := pairs.filter (λ p => p.fst = color)
  if matching_pair.length = 1 then
   let n := matching_pair.head!.snd * 2
   (n / total_shoes) * ((n / 2) / (total_shoes - 1))
  else 0

def prob_total (pairs : List (String × ℕ)) : ℚ :=
  (prob_same_color "black" pairs) + (prob_same_color "brown" pairs) + (prob_same_color "gray" pairs)

theorem sue_shoes_probability :
  prob_total sueShoes = 31 / 138 := by
  sorry

end sue_shoes_probability_l908_90839


namespace fraction_multiplication_l908_90857

theorem fraction_multiplication :
  (3 / 4 : ℚ) * (1 / 2) * (2 / 5) * 5000 = 750 :=
by
  norm_num
  done

end fraction_multiplication_l908_90857


namespace ordered_pairs_m_n_l908_90889

theorem ordered_pairs_m_n :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ (p.1 ^ 2 - p.2 ^ 2 = 72)) ∧ s.card = 3 :=
by
  sorry

end ordered_pairs_m_n_l908_90889


namespace p_plus_q_l908_90820

-- Define the circles w1 and w2
def circle1 (x y : ℝ) := x^2 + y^2 + 10*x - 20*y - 77 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 10*x - 20*y + 193 = 0

-- Define the line condition
def line (a x y : ℝ) := y = a * x

-- Prove that p + q = 85, where m^2 = p / q and m is the smallest positive a
theorem p_plus_q : ∃ p q : ℕ, (p.gcd q = 1) ∧ (m^2 = (p : ℝ)/(q : ℝ)) ∧ (p + q = 85) :=
  sorry

end p_plus_q_l908_90820


namespace attendees_count_l908_90827

def n_students_seated : ℕ := 300
def n_students_standing : ℕ := 25
def n_teachers_seated : ℕ := 30

def total_attendees : ℕ :=
  n_students_seated + n_students_standing + n_teachers_seated

theorem attendees_count :
  total_attendees = 355 := by
  sorry

end attendees_count_l908_90827


namespace hosting_schedules_count_l908_90860

theorem hosting_schedules_count :
  let n_universities := 6
  let n_years := 8
  let total_ways := 6 * 5 * 4^6
  let excluding_one := 6 * 5 * 4 * 3^6
  let excluding_two := 15 * 4 * 3 * 2^6
  let excluding_three := 20 * 3 * 2 * 1^6
  total_ways - excluding_one + excluding_two - excluding_three = 46080 := 
by
  sorry

end hosting_schedules_count_l908_90860


namespace neg_ln_gt_zero_l908_90813

theorem neg_ln_gt_zero {x : ℝ} : (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ ∃ x : ℝ, Real.log (x^2 + 1) ≤ 0 := by
  sorry

end neg_ln_gt_zero_l908_90813


namespace student_survey_l908_90842

-- Define the conditions given in the problem
theorem student_survey (S F : ℝ) (h1 : F = 25 + 65) (h2 : F = 0.45 * S) : S = 200 :=
by
  sorry

end student_survey_l908_90842


namespace dreams_ratio_l908_90853

variable (N : ℕ) (D_total : ℕ) (D_per_day : ℕ)

-- Conditions
def days_per_year : Prop := N = 365
def dreams_per_day : Prop := D_per_day = 4
def total_dreams : Prop := D_total = 4380

-- Derived definitions
def dreams_this_year := D_per_day * N
def dreams_last_year := D_total - dreams_this_year

-- Theorem to prove
theorem dreams_ratio 
  (h1 : days_per_year N)
  (h2 : dreams_per_day D_per_day)
  (h3 : total_dreams D_total)
  : dreams_last_year N D_total D_per_day / dreams_this_year N D_per_day = 2 :=
by
  sorry

end dreams_ratio_l908_90853


namespace find_number_l908_90822

theorem find_number : ∃ (x : ℤ), 45 + 3 * x = 72 ∧ x = 9 := by
  sorry

end find_number_l908_90822


namespace correct_answer_is_B_l908_90858

def lack_of_eco_friendly_habits : Prop := true
def major_global_climate_change_cause (s : String) : Prop :=
  s = "cause"

theorem correct_answer_is_B :
  major_global_climate_change_cause "cause" ∧ lack_of_eco_friendly_habits → "B" = "cause" :=
by
  sorry

end correct_answer_is_B_l908_90858


namespace express_in_scientific_notation_l908_90835

theorem express_in_scientific_notation 
  (A : 149000000 = 149 * 10^6)
  (B : 149000000 = 1.49 * 10^8)
  (C : 149000000 = 14.9 * 10^7)
  (D : 149000000 = 1.5 * 10^8) :
  149000000 = 1.49 * 10^8 := 
by
  sorry

end express_in_scientific_notation_l908_90835


namespace ab_times_65_eq_48ab_l908_90810

theorem ab_times_65_eq_48ab (a b : ℕ) (h_ab : 0 ≤ a ∧ a < 10) (h_b : 0 ≤ b ∧ b < 10) :
  (10 * a + b) * 65 = 4800 + 10 * a + b ↔ 10 * a + b = 75 := by
sorry

end ab_times_65_eq_48ab_l908_90810


namespace max_ab_l908_90830

theorem max_ab (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 3 ≤ a + b ∧ a + b ≤ 4) : ab ≤ 15 / 4 :=
sorry

end max_ab_l908_90830


namespace inequality_solution_l908_90875

theorem inequality_solution (x : ℤ) : (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 → x ≥ -5 := 
by
  sorry

end inequality_solution_l908_90875


namespace smallest_integer_in_ratio_l908_90828

theorem smallest_integer_in_ratio {a b c : ℕ} (h1 : a = 2 * b / 3) (h2 : c = 5 * b / 3) (h3 : a + b + c = 60) : b = 12 := 
  sorry

end smallest_integer_in_ratio_l908_90828


namespace cakesServedDuringDinner_today_is_6_l908_90803

def cakesServedDuringDinner (x : ℕ) : Prop :=
  5 + x + 3 = 14

theorem cakesServedDuringDinner_today_is_6 : cakesServedDuringDinner 6 :=
by
  unfold cakesServedDuringDinner
  -- The proof is omitted
  sorry

end cakesServedDuringDinner_today_is_6_l908_90803


namespace pupils_sent_up_exam_l908_90833

theorem pupils_sent_up_exam (average_marks : ℕ) (specific_scores : List ℕ) (new_average : ℕ) : 
  (average_marks = 39) → 
  (specific_scores = [25, 12, 15, 19]) → 
  (new_average = 44) → 
  ∃ n : ℕ, (n > 4) ∧ (average_marks * n) = 39 * n ∧ ((39 * n - specific_scores.sum) / (n - specific_scores.length)) = new_average →
  n = 21 :=
by
  intros h_avg h_scores h_new_avg
  sorry

end pupils_sent_up_exam_l908_90833


namespace inequality_abc_l908_90816

theorem inequality_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  abs (b / a - b / c) + abs (c / a - c / b) + abs (b * c + 1) > 1 :=
by
  sorry

end inequality_abc_l908_90816


namespace no_int_b_exists_l908_90874

theorem no_int_b_exists (k n a : ℕ) (hk3 : k ≥ 3) (hn3 : n ≥ 3) (hk_odd : k % 2 = 1) (hn_odd : n % 2 = 1)
  (ha1 : a ≥ 1) (hka : k ∣ (2^a + 1)) (hna : n ∣ (2^a - 1)) :
  ¬ ∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
sorry

end no_int_b_exists_l908_90874


namespace stuffed_dogs_count_l908_90869

theorem stuffed_dogs_count (D : ℕ) (h1 : 14 + D % 7 = 0) : D = 7 :=
by {
  sorry
}

end stuffed_dogs_count_l908_90869


namespace correct_equation_l908_90812

theorem correct_equation (x : ℝ) :
  232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l908_90812


namespace difference_of_numbers_l908_90807

theorem difference_of_numbers : 
  ∃ (L S : ℕ), L = 1631 ∧ L = 6 * S + 35 ∧ L - S = 1365 := 
by
  sorry

end difference_of_numbers_l908_90807


namespace min_sum_of_gcd_and_lcm_eq_three_times_sum_l908_90870

theorem min_sum_of_gcd_and_lcm_eq_three_times_sum (a b d : ℕ) (h1 : d = Nat.gcd a b)
  (h2 : Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) :
  a + b = 12 :=
by
sorry

end min_sum_of_gcd_and_lcm_eq_three_times_sum_l908_90870


namespace x_squared_y_cubed_eq_200_l908_90878

theorem x_squared_y_cubed_eq_200 (x y : ℕ) (h : 2^x * 9^y = 200) : x^2 * y^3 = 200 := by
  sorry

end x_squared_y_cubed_eq_200_l908_90878


namespace radio_loss_percentage_l908_90838

theorem radio_loss_percentage (CP SP : ℝ) (h_CP : CP = 2400) (h_SP : SP = 2100) :
  ((CP - SP) / CP) * 100 = 12.5 :=
by
  -- Given cost price
  have h_CP : CP = 2400 := h_CP
  -- Given selling price
  have h_SP : SP = 2100 := h_SP
  sorry

end radio_loss_percentage_l908_90838


namespace third_number_l908_90808

theorem third_number (x : ℝ) 
    (h : 217 + 2.017 + 2.0017 + x = 221.2357) : 
    x = 0.217 :=
sorry

end third_number_l908_90808


namespace maximum_p_l908_90854

noncomputable def p (a b c : ℝ) : ℝ :=
  (2 / (a ^ 2 + 1)) - (2 / (b ^ 2 + 1)) + (3 / (c ^ 2 + 1))

theorem maximum_p (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : abc + a + c = b) : 
  p a b c ≤ 10 / 3 ∧ ∃ a b c, abc + a + c = b ∧ p a b c = 10 / 3 :=
sorry

end maximum_p_l908_90854


namespace simplify_expression_l908_90829

variable (x : Int)

theorem simplify_expression : 3 * x + 5 * x + 7 * x = 15 * x :=
  by
  sorry

end simplify_expression_l908_90829
