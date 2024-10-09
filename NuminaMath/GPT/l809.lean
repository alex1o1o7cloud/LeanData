import Mathlib

namespace sum_2_75_0_003_0_158_l809_80906

theorem sum_2_75_0_003_0_158 : 2.75 + 0.003 + 0.158 = 2.911 :=
by
  -- Lean proof goes here  
  sorry

end sum_2_75_0_003_0_158_l809_80906


namespace tree_height_l809_80949

theorem tree_height (B h : ℕ) (H : ℕ) (h_eq : h = 16) (B_eq : B = 12) (L : ℕ) (L_def : L ^ 2 = B ^ 2 + h ^ 2) (H_def : H = h + L) :
    H = 36 := by
  -- We do not need to provide the proof steps as per the instructions
  sorry

end tree_height_l809_80949


namespace seq_problem_l809_80953

theorem seq_problem (a : ℕ → ℚ) (d : ℚ) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d )
 (h1 : a 1 = 2)
 (h_geom : (a 1 - 1) * (a 5 + 5) = (a 3)^2) :
  a 2017 = 1010 := 
sorry

end seq_problem_l809_80953


namespace max_discount_rate_l809_80967

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l809_80967


namespace similar_triangles_side_length_l809_80932

theorem similar_triangles_side_length
  (A1 A2 : ℕ) (k : ℕ) (h1 : A1 - A2 = 18)
  (h2 : A1 = k^2 * A2) (h3 : ∃ n : ℕ, A2 = n)
  (s : ℕ) (h4 : s = 3) :
  s * k = 6 :=
by
  sorry

end similar_triangles_side_length_l809_80932


namespace inequality_solution_l809_80966

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end inequality_solution_l809_80966


namespace pat_oj_consumption_l809_80936

def initial_oj : ℚ := 3 / 4
def alex_fraction : ℚ := 1 / 2
def pat_fraction : ℚ := 1 / 3

theorem pat_oj_consumption : pat_fraction * (initial_oj * (1 - alex_fraction)) = 1 / 8 := by
  -- This will be the proof part which can be filled later
  sorry

end pat_oj_consumption_l809_80936


namespace distinct_orders_scoops_l809_80980

-- Conditions
def total_scoops : ℕ := 4
def chocolate_scoops : ℕ := 2
def vanilla_scoops : ℕ := 1
def strawberry_scoops : ℕ := 1

-- Problem statement
theorem distinct_orders_scoops :
  (Nat.factorial total_scoops) / ((Nat.factorial chocolate_scoops) * (Nat.factorial vanilla_scoops) * (Nat.factorial strawberry_scoops)) = 12 := by
  sorry

end distinct_orders_scoops_l809_80980


namespace x_plus_y_eq_3012_plus_pi_div_2_l809_80901

theorem x_plus_y_eq_3012_plus_pi_div_2
  (x y : ℝ)
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 :=
sorry

end x_plus_y_eq_3012_plus_pi_div_2_l809_80901


namespace boat_speed_l809_80983

theorem boat_speed (v : ℝ) (h1 : 5 + v = 30) : v = 25 :=
by 
  -- Solve for the speed of the second boat
  sorry

end boat_speed_l809_80983


namespace triangle_existence_l809_80933

theorem triangle_existence (n : ℕ) (h : 2 * n > 0) (segments : Finset (ℕ × ℕ))
  (h_segments : segments.card = n^2 + 1)
  (points_in_segment : ∀ {a b : ℕ}, (a, b) ∈ segments → a < 2 * n ∧ b < 2 * n) :
  ∃ x y z, x < 2 * n ∧ y < 2 * n ∧ z < 2 * n ∧ (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  ((x, y) ∈ segments ∨ (y, x) ∈ segments) ∧
  ((y, z) ∈ segments ∨ (z, y) ∈ segments) ∧
  ((z, x) ∈ segments ∨ (x, z) ∈ segments) :=
by
  sorry

end triangle_existence_l809_80933


namespace existence_of_five_regular_polyhedra_l809_80944

def regular_polyhedron (n m : ℕ) : Prop :=
  n ≥ 3 ∧ m ≥ 3 ∧ (2 / m + 2 / n > 1)

theorem existence_of_five_regular_polyhedra :
  ∃ (n m : ℕ), regular_polyhedron n m → 
    (n = 3 ∧ m = 3 ∨ 
     n = 4 ∧ m = 3 ∨ 
     n = 3 ∧ m = 4 ∨ 
     n = 5 ∧ m = 3 ∨ 
     n = 3 ∧ m = 5) :=
by
  sorry

end existence_of_five_regular_polyhedra_l809_80944


namespace fraction_subtraction_simplify_l809_80964

noncomputable def fraction_subtraction : ℚ :=
  (12 / 25) - (3 / 75)

theorem fraction_subtraction_simplify : fraction_subtraction = (11 / 25) :=
  by
    -- Proof goes here
    sorry

end fraction_subtraction_simplify_l809_80964


namespace find_a3_l809_80908

-- Define the geometric sequence and its properties.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_GeoSeq : is_geometric_sequence a q)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 9)

-- Define what we need to prove
theorem find_a3 : a 3 = 3 :=
sorry

end find_a3_l809_80908


namespace probability_at_least_seven_heads_or_tails_l809_80914

open Nat

-- Define the probability of getting at least seven heads or tails in eight coin flips
theorem probability_at_least_seven_heads_or_tails :
  let total_outcomes := 2^8
  let favorable_outcomes := (choose 8 7) + (choose 8 7) + 1 + 1
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 9 / 128 := by
  sorry

end probability_at_least_seven_heads_or_tails_l809_80914


namespace zero_of_function_l809_80994

theorem zero_of_function : ∃ x : ℝ, (x + 1)^2 = 0 :=
by
  use -1
  sorry

end zero_of_function_l809_80994


namespace tan_two_beta_l809_80975

variables {α β : Real}

theorem tan_two_beta (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 7) : Real.tan (2 * β) = -3 / 4 :=
by
  sorry

end tan_two_beta_l809_80975


namespace problem1_problem2_problem3_problem4_l809_80973

theorem problem1 (h : Real.cos 75 * Real.sin 75 = 1 / 2) : False :=
by
  sorry

theorem problem2 : (1 + Real.tan 15) / (1 - Real.tan 15) = Real.sqrt 3 :=
by
  sorry

theorem problem3 : Real.tan 20 + Real.tan 25 + Real.tan 20 * Real.tan 25 = 1 :=
by
  sorry

theorem problem4 (θ : Real) (h1 : Real.sin (2 * θ) ≠ 0) : (1 / Real.tan θ - 1 / Real.tan (2 * θ) = 1 / Real.sin (2 * θ)) :=
by
  sorry

end problem1_problem2_problem3_problem4_l809_80973


namespace average_visitors_on_Sundays_l809_80903

theorem average_visitors_on_Sundays (S : ℕ) 
  (h1 : 30 % 7 = 2)  -- The month begins with a Sunday
  (h2 : 25 = 30 - 5)  -- The month has 25 non-Sundays
  (h3 : (120 * 25) = 3000) -- Total visitors on non-Sundays
  (h4 : (125 * 30) = 3750) -- Total visitors for the month
  (h5 : 5 * 30 > 0) -- There are a positive number of Sundays
  : S = 150 :=
by
  sorry

end average_visitors_on_Sundays_l809_80903


namespace min_length_PQ_l809_80924

noncomputable def minimum_length (a : ℝ) : ℝ :=
  let x := 2 * a
  let y := a + 2
  let d := |2 * 2 - 2 * 0 + 4| / Real.sqrt (1^2 + (-2)^2)
  let r := Real.sqrt 5
  d - r

theorem min_length_PQ : ∀ (a : ℝ), P ∈ {P : ℝ × ℝ | (P.1 - 2)^2 + P.2^2 = 5} ∧ Q = (2 * a, a + 2) →
  minimum_length a = 3 * Real.sqrt 5 / 5 :=
by
  intro a
  intro h
  rcases h with ⟨hP, hQ⟩
  sorry

end min_length_PQ_l809_80924


namespace actual_time_when_watch_shows_8_PM_l809_80912

-- Definitions based on the problem's conditions
def initial_time := 8  -- 8:00 AM
def incorrect_watch_time := 14 * 60 + 42  -- 2:42 PM converted to minutes
def actual_time := 15 * 60  -- 3:00 PM converted to minutes
def target_watch_time := 20 * 60  -- 8:00 PM converted to minutes

-- Define to calculate the rate of time loss
def time_loss_rate := (actual_time - incorrect_watch_time) / (actual_time - initial_time * 60)

-- Hypothesis that the watch loses time at a constant rate
axiom constant_rate : ∀ t, t >= initial_time * 60 ∧ t <= actual_time → (t * time_loss_rate) = (actual_time - incorrect_watch_time)

-- Define the target time based on watch reading 8:00 PM
noncomputable def target_actual_time := target_watch_time / time_loss_rate

-- Main theorem: Prove that given the conditions, the target actual time is 8:32 PM
theorem actual_time_when_watch_shows_8_PM : target_actual_time = (20 * 60 + 32) :=
sorry

end actual_time_when_watch_shows_8_PM_l809_80912


namespace factorization_of_polynomial_l809_80963

noncomputable def p (x : ℤ) : ℤ := x^15 + x^10 + x^5 + 1
noncomputable def f (x : ℤ) : ℤ := x^3 + x^2 + x + 1
noncomputable def g (x : ℤ) : ℤ := x^12 - x^11 + x^9 - x^8 + x^6 - x^5 + x^3 - x^2 + x - 1

theorem factorization_of_polynomial : ∀ x : ℤ, p x = f x * g x :=
by sorry

end factorization_of_polynomial_l809_80963


namespace find_a_value_l809_80970

theorem find_a_value : 
  (∀ x, (3 * (x - 2) - 4 * (x - 5 / 4) = 0) ↔ ( ∃ a, ((2 * x - a) / 3 - (x - a) / 2 = x - 1) ∧ a = -11 )) := sorry

end find_a_value_l809_80970


namespace Kyler_wins_1_game_l809_80916

theorem Kyler_wins_1_game
  (peter_wins : ℕ)
  (peter_losses : ℕ)
  (emma_wins : ℕ)
  (emma_losses : ℕ)
  (kyler_losses : ℕ)
  (total_games : ℕ)
  (kyler_wins : ℕ)
  (htotal : total_games = (peter_wins + peter_losses + emma_wins + emma_losses + kyler_wins + kyler_losses) / 2)
  (hpeter : peter_wins = 4 ∧ peter_losses = 2)
  (hemma : emma_wins = 3 ∧ emma_losses = 3)
  (hkyler_losses : kyler_losses = 3)
  (htotal_wins_losses : total_games = peter_wins + emma_wins + kyler_wins) : kyler_wins = 1 :=
by
  sorry

end Kyler_wins_1_game_l809_80916


namespace total_students_sampled_l809_80942

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l809_80942


namespace max_visible_unit_cubes_l809_80990

def cube_size := 11
def total_unit_cubes := cube_size ^ 3

def visible_unit_cubes (n : ℕ) : ℕ :=
  (n * n) + (n * (n - 1)) + ((n - 1) * (n - 1))

theorem max_visible_unit_cubes : 
  visible_unit_cubes cube_size = 331 := by
  sorry

end max_visible_unit_cubes_l809_80990


namespace christina_walking_speed_l809_80968

noncomputable def christina_speed : ℕ :=
  let distance_between := 270
  let jack_speed := 4
  let lindy_total_distance := 240
  let lindy_speed := 8
  let meeting_time := lindy_total_distance / lindy_speed
  let jack_covered := jack_speed * meeting_time
  let remaining_distance := distance_between - jack_covered
  remaining_distance / meeting_time

theorem christina_walking_speed : christina_speed = 5 := by
  -- Proof will be provided here to verify the theorem, but for now, we use sorry to skip it
  sorry

end christina_walking_speed_l809_80968


namespace chocolate_chip_cookie_count_l809_80997

-- Let cookies_per_bag be the number of cookies in each bag
def cookies_per_bag : ℕ := 5

-- Let oatmeal_cookies be the number of oatmeal cookies
def oatmeal_cookies : ℕ := 2

-- Let num_baggies be the number of baggies
def num_baggies : ℕ := 7

-- Define the total number of cookies as num_baggies * cookies_per_bag
def total_cookies : ℕ := num_baggies * cookies_per_bag

-- Define the number of chocolate chip cookies as total_cookies - oatmeal_cookies
def chocolate_chip_cookies : ℕ := total_cookies - oatmeal_cookies

-- Prove that the number of chocolate chip cookies is 33
theorem chocolate_chip_cookie_count : chocolate_chip_cookies = 33 := by
  sorry

end chocolate_chip_cookie_count_l809_80997


namespace exactly_one_divisible_by_4_l809_80984

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l809_80984


namespace find_m_l809_80982

theorem find_m (m : ℝ) :
  (∀ x y, x + (m^2 - m) * y = 4 * m - 1 → ∀ x y, 2 * x - y - 5 = 0 → (-1 / (m^2 - m)) = -1 / 2) → 
  (m = -1 ∨ m = 2) :=
sorry

end find_m_l809_80982


namespace charlie_fewer_games_than_dana_l809_80934

theorem charlie_fewer_games_than_dana
  (P D C Ph : ℕ)
  (h1 : P = D + 5)
  (h2 : C < D)
  (h3 : Ph = C + 3)
  (h4 : Ph = 12)
  (h5 : P = Ph + 4) :
  D - C = 2 :=
by
  sorry

end charlie_fewer_games_than_dana_l809_80934


namespace hamburger_combinations_l809_80999

def number_of_condiments := 8
def condiment_combinations := 2 ^ number_of_condiments
def number_of_meat_patties := 4
def total_hamburgers := number_of_meat_patties * condiment_combinations

theorem hamburger_combinations :
  total_hamburgers = 1024 :=
by
  sorry

end hamburger_combinations_l809_80999


namespace range_of_expressions_l809_80904

theorem range_of_expressions (x y : ℝ) (h1 : 30 < x ∧ x < 42) (h2 : 16 < y ∧ y < 24) :
  46 < x + y ∧ x + y < 66 ∧ -18 < x - 2 * y ∧ x - 2 * y < 10 ∧ (5 / 4) < (x / y) ∧ (x / y) < (21 / 8) :=
sorry

end range_of_expressions_l809_80904


namespace product_of_distinct_integers_l809_80930

def is2008thPower (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2008

theorem product_of_distinct_integers {x y z : ℕ} (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x)
  (h4 : y = (x + z) / 2) (h5 : x > 0) (h6 : y > 0) (h7 : z > 0) 
  : is2008thPower (x * y * z) :=
  sorry

end product_of_distinct_integers_l809_80930


namespace find_f_x_minus_1_l809_80978

theorem find_f_x_minus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x ^ 2 + 2 * x) :
  ∀ x : ℤ, f (x - 1) = x ^ 2 - 2 * x :=
by
  sorry

end find_f_x_minus_1_l809_80978


namespace line_intersects_circle_l809_80913

theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (h_outside : a^2 + b^2 > r^2) :
  ∃ x y : ℝ, (x^2 + y^2 = r^2) ∧ (a * x + b * y = r^2) :=
sorry

end line_intersects_circle_l809_80913


namespace least_remaining_marbles_l809_80947

/-- 
There are 60 identical marbles forming a tetrahedral pile.
The formula for the number of marbles in a tetrahedral pile up to the k-th level is given by:
∑_(i=1)^k (i * (i + 1)) / 6 = k * (k + 1) * (k + 2) / 6.

We must show that the least number of remaining marbles when 60 marbles are used to form the pile is 4.
-/
theorem least_remaining_marbles : ∃ k : ℕ, (60 - k * (k + 1) * (k + 2) / 6) = 4 :=
by
  sorry

end least_remaining_marbles_l809_80947


namespace recipe_serves_correctly_l809_80910

theorem recipe_serves_correctly:
  ∀ (cream_fat_per_cup : ℝ) (cream_amount_cup : ℝ) (fat_per_serving : ℝ) (total_servings: ℝ),
    cream_fat_per_cup = 88 →
    cream_amount_cup = 0.5 →
    fat_per_serving = 11 →
    total_servings = (cream_amount_cup * cream_fat_per_cup) / fat_per_serving →
    total_servings = 4 :=
by
  intros cream_fat_per_cup cream_amount_cup fat_per_serving total_servings
  intros hcup hccup hfserv htserv
  sorry

end recipe_serves_correctly_l809_80910


namespace intersecting_points_of_curves_l809_80956

theorem intersecting_points_of_curves :
  (∀ x y, (y = 2 * x^3 + x^2 - 5 * x + 2) ∧ (y = 3 * x^2 + 6 * x - 4) → 
   (x = -1 ∧ y = -7) ∨ (x = 3 ∧ y = 41)) := sorry

end intersecting_points_of_curves_l809_80956


namespace sum_6n_is_correct_l809_80971

theorem sum_6n_is_correct {n : ℕ} (h : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by sorry

end sum_6n_is_correct_l809_80971


namespace color_dot_figure_l809_80938

-- Definitions reflecting the problem conditions
def num_colors : ℕ := 3
def first_triangle_coloring_ways : ℕ := 6
def subsequent_triangle_coloring_ways : ℕ := 3
def additional_dot_coloring_ways : ℕ := 2

-- The theorem stating the required proof
theorem color_dot_figure : first_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           subsequent_triangle_coloring_ways * 
                           additional_dot_coloring_ways = 108 := by
sorry

end color_dot_figure_l809_80938


namespace surface_area_of_sphere_containing_prism_l809_80988

-- Assume the necessary geometric context and definitions are available.
def rightSquarePrism (a h : ℝ) (V : ℝ) := 
  a^2 * h = V

theorem surface_area_of_sphere_containing_prism 
  (a h V : ℝ) (S : ℝ) (π := Real.pi)
  (prism_on_sphere : ∀ (prism : rightSquarePrism a h V), True)
  (height_eq_4 : h = 4) 
  (volume_eq_16 : V = 16) :
  S = 4 * π * 24 :=
by
  -- proof steps would go here
  sorry

end surface_area_of_sphere_containing_prism_l809_80988


namespace avg_equivalence_l809_80957

-- Definition of binary average [a, b]
def avg2 (a b : ℤ) : ℤ := (a + b) / 2

-- Definition of ternary average {a, b, c}
def avg3 (a b c : ℤ) : ℤ := (a + b + c) / 3

-- Lean statement for proving the given problem
theorem avg_equivalence : avg3 (avg3 2 2 (-1)) (avg2 3 (-1)) 1 = 1 := by
  sorry

end avg_equivalence_l809_80957


namespace factorization_problem_l809_80959

theorem factorization_problem 
    (a m n b : ℝ)
    (h1 : (x + 2) * (x + 4) = x^2 + a * x + m)
    (h2 : (x + 1) * (x + 9) = x^2 + n * x + b) :
    (x + 3) * (x + 3) = x^2 + a * x + b :=
by
  sorry

end factorization_problem_l809_80959


namespace sum_of_digits_18_to_21_l809_80900

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_18_to_21 : 
  (sum_digits 18 + sum_digits 19 + sum_digits 20 + sum_digits 21) = 24 := 
by 
  sorry

end sum_of_digits_18_to_21_l809_80900


namespace largest_common_remainder_l809_80948

theorem largest_common_remainder : 
  ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r = 4) := 
by
  sorry

end largest_common_remainder_l809_80948


namespace e_exp_ax1_ax2_gt_two_l809_80981

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem e_exp_ax1_ax2_gt_two {a x1 x2 : ℝ} (h : a ≠ 0) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (hx : x1 < x2) : 
  Real.exp (a * x1) + Real.exp (a * x2) > 2 :=
sorry

end e_exp_ax1_ax2_gt_two_l809_80981


namespace song_book_cost_correct_l809_80945

/-- Define the constants for the problem. -/
def clarinet_cost : ℝ := 130.30
def pocket_money : ℝ := 12.32
def total_spent : ℝ := 141.54

/-- Prove the cost of the song book. -/
theorem song_book_cost_correct :
  (total_spent - clarinet_cost) = 11.24 :=
by
  sorry

end song_book_cost_correct_l809_80945


namespace arithmetic_sequence_a5_l809_80985

theorem arithmetic_sequence_a5 (a_n : ℕ → ℝ) 
  (h_arith : ∀ n, a_n (n+1) - a_n n = a_n (n+2) - a_n (n+1))
  (h_condition : a_n 1 + a_n 9 = 10) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_l809_80985


namespace average_GPA_of_whole_class_l809_80925

variable (n : ℕ)

def GPA_first_group : ℕ := 54 * (n / 3)
def GPA_second_group : ℕ := 45 * (2 * n / 3)
def total_GPA : ℕ := GPA_first_group n + GPA_second_group n

theorem average_GPA_of_whole_class : total_GPA n / n = 48 := by
  sorry

end average_GPA_of_whole_class_l809_80925


namespace find_values_l809_80987

theorem find_values (a b: ℝ) (h1: a > b) (h2: b > 1)
  (h3: Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h4: a^b = b^a) :
  a = 4 ∧ b = 2 := 
sorry

end find_values_l809_80987


namespace odds_against_C_win_l809_80989

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C_win (pA pB : ℚ) (hA : pA = 1/5) (hB : pB = 2/3) :
  odds_against_winning (1 - pA - pB) = 13 / 2 :=
by
  sorry

end odds_against_C_win_l809_80989


namespace locus_of_midpoint_of_square_l809_80962

theorem locus_of_midpoint_of_square (a : ℝ) (x y : ℝ) (h1 : x^2 + y^2 = 4 * a^2) :
  (∃ X Y : ℝ, 2 * X = x ∧ 2 * Y = y ∧ X^2 + Y^2 = a^2) :=
by {
  -- No proof is required, so we use 'sorry' here
  sorry
}

end locus_of_midpoint_of_square_l809_80962


namespace determine_expr_l809_80919

noncomputable def expr (a b c d : ℝ) : ℝ :=
  (1 + a + a * b) / (1 + a + a * b + a * b * c) +
  (1 + b + b * c) / (1 + b + b * c + b * c * d) +
  (1 + c + c * d) / (1 + c + c * d + c * d * a) +
  (1 + d + d * a) / (1 + d + d * a + d * a * b)

theorem determine_expr (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  expr a b c d = 2 :=
sorry

end determine_expr_l809_80919


namespace ratio_cost_to_marked_price_l809_80998

variables (x : ℝ) (marked_price : ℝ) (selling_price : ℝ) (cost_price : ℝ)

theorem ratio_cost_to_marked_price :
  (selling_price = marked_price - 1/4 * marked_price) →
  (cost_price = 2/3 * selling_price) →
  (cost_price / marked_price = 1/2) :=
by
  sorry

end ratio_cost_to_marked_price_l809_80998


namespace num_ordered_triples_l809_80911

/-
Let Q be a right rectangular prism with integral side lengths a, b, and c such that a ≤ b ≤ c, and b = 2023.
A plane parallel to one of the faces of Q cuts Q into two prisms, one of which is similar to Q, and both have nonzero volume.
Prove that the number of ordered triples (a, b, c) such that b = 2023 is 7.
-/

theorem num_ordered_triples (a c : ℕ) (h : a ≤ 2023 ∧ 2023 ≤ c) (ac_eq_2023_squared : a * c = 2023^2) :
  ∃ count, count = 7 :=
by {
  sorry
}

end num_ordered_triples_l809_80911


namespace hot_dogs_per_pack_l809_80939

-- Define the givens / conditions
def total_hot_dogs : ℕ := 36
def buns_pack_size : ℕ := 9
def same_quantity (h : ℕ) (b : ℕ) := h = b

-- State the theorem to be proven
theorem hot_dogs_per_pack : ∃ h : ℕ, (total_hot_dogs / h = buns_pack_size) ∧ same_quantity (total_hot_dogs / h) (total_hot_dogs / buns_pack_size) := 
sorry

end hot_dogs_per_pack_l809_80939


namespace proposition2_and_4_correct_l809_80955

theorem proposition2_and_4_correct (a b : ℝ) : 
  (a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧ 
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → a^2 + b^2 = 9) :=
by
  sorry

end proposition2_and_4_correct_l809_80955


namespace necessary_but_not_sufficient_l809_80960

def p (a : ℝ) : Prop := (a - 1) * (a - 2) = 0
def q (a : ℝ) : Prop := a = 1

theorem necessary_but_not_sufficient (a : ℝ) : 
  (q a → p a) ∧ (p a → q a → False) :=
by
  sorry

end necessary_but_not_sufficient_l809_80960


namespace verify_salary_problem_l809_80991

def salary_problem (W : ℕ) (S_old : ℕ) (S_new : ℕ := 780) (n : ℕ := 9) : Prop :=
  (W + S_old) / n = 430 ∧ (W + S_new) / n = 420 → S_old = 870

theorem verify_salary_problem (W S_old : ℕ) (h1 : (W + S_old) / 9 = 430) (h2 : (W + 780) / 9 = 420) : S_old = 870 :=
by {
  sorry
}

end verify_salary_problem_l809_80991


namespace find_minimum_n_l809_80972

noncomputable def a_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ := 1 / 2 * (3 ^ n - 1)

theorem find_minimum_n (S_n : ℕ → ℕ) (n : ℕ) :
  (3^n - 1) / 2 > 1000 → n = 7 := 
sorry

end find_minimum_n_l809_80972


namespace instantaneous_velocity_at_3_l809_80907

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (deriv (motion_equation) 3 = 5) :=
by
  sorry

end instantaneous_velocity_at_3_l809_80907


namespace find_multiplier_l809_80918

theorem find_multiplier :
  ∀ (x n : ℝ), (x = 5) → (x * n = (16 - x) + 4) → n = 3 :=
by
  intros x n hx heq
  sorry

end find_multiplier_l809_80918


namespace herman_days_per_week_l809_80905

-- Defining the given conditions as Lean definitions
def total_meals : ℕ := 4
def cost_per_meal : ℕ := 4
def total_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Calculating derived facts based on given conditions
def cost_per_day : ℕ := total_meals * cost_per_meal
def cost_per_week : ℕ := total_cost / total_weeks

-- Our main theorem that states Herman buys breakfast combos 5 days per week
theorem herman_days_per_week : cost_per_week / cost_per_day = 5 :=
by
  -- Skipping the proof
  sorry

end herman_days_per_week_l809_80905


namespace alice_paper_cranes_l809_80940

theorem alice_paper_cranes (T : ℕ)
  (h1 : T / 2 - T / 10 = 400) : T = 1000 :=
sorry

end alice_paper_cranes_l809_80940


namespace smaller_sphere_radius_l809_80992

theorem smaller_sphere_radius (R x : ℝ) (h1 : (4/3) * Real.pi * R^3 = (4/3) * Real.pi * x^3 + (4/3) * Real.pi * (2 * x)^3) 
  (h2 : ∀ r₁ r₂ : ℝ, r₁ / r₂ = 1 / 2 → r₁ = x ∧ r₂ = 2 * x) : x = R / 3 :=
by 
  sorry

end smaller_sphere_radius_l809_80992


namespace minimum_value_l809_80935

theorem minimum_value (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81 / x^4 = 24 := 
sorry

end minimum_value_l809_80935


namespace eli_age_difference_l809_80928

theorem eli_age_difference (kaylin_age : ℕ) (freyja_age : ℕ) (sarah_age : ℕ) (eli_age : ℕ) 
  (H1 : kaylin_age = 33)
  (H2 : freyja_age = 10)
  (H3 : kaylin_age + 5 = sarah_age)
  (H4 : sarah_age = 2 * eli_age) :
  eli_age - freyja_age = 9 := 
sorry

end eli_age_difference_l809_80928


namespace Euler_theorem_l809_80951

theorem Euler_theorem {m a : ℕ} (hm : m ≥ 1) (h_gcd : Nat.gcd a m = 1) : a ^ Nat.totient m ≡ 1 [MOD m] :=
by
  sorry

end Euler_theorem_l809_80951


namespace find_tax_rate_l809_80979

variable (total_spent : ℝ) (sales_tax : ℝ) (tax_free_cost : ℝ) (taxable_items_cost : ℝ) 
variable (T : ℝ)

theorem find_tax_rate (h1 : total_spent = 25) 
                      (h2 : sales_tax = 0.30)
                      (h3 : tax_free_cost = 21.7)
                      (h4 : taxable_items_cost = total_spent - tax_free_cost - sales_tax)
                      (h5 : sales_tax = (T / 100) * taxable_items_cost) :
  T = 10 := 
sorry

end find_tax_rate_l809_80979


namespace yoga_studio_women_count_l809_80977

theorem yoga_studio_women_count :
  ∃ W : ℕ, 
  (8 * 190) + (W * 120) = 14 * 160 ∧ W = 6 :=
by 
  existsi (6);
  sorry

end yoga_studio_women_count_l809_80977


namespace no_real_solutions_for_equation_l809_80976

theorem no_real_solutions_for_equation (x : ℝ) :
  y = 3 * x ∧ y = (x^3 - 8) / (x - 2) → false :=
by {
  sorry
}

end no_real_solutions_for_equation_l809_80976


namespace find_number_l809_80952

theorem find_number (n : ℝ) : (2629.76 / n = 528.0642570281125) → n = 4.979 :=
by
  intro h
  sorry

end find_number_l809_80952


namespace time_to_cover_length_l809_80902

def escalator_speed : ℝ := 8  -- The speed of the escalator in feet per second
def person_speed : ℝ := 2     -- The speed of the person in feet per second
def escalator_length : ℝ := 160 -- The length of the escalator in feet

theorem time_to_cover_length : 
  (escalator_length / (escalator_speed + person_speed) = 16) :=
by 
  sorry

end time_to_cover_length_l809_80902


namespace Joe_total_income_l809_80931

theorem Joe_total_income : 
  (∃ I : ℝ, 0.1 * 1000 + 0.2 * 3000 + 0.3 * (I - 500 - 4000) = 848 ∧ I - 500 > 4000) → I = 4993.33 :=
by
  sorry

end Joe_total_income_l809_80931


namespace negation_of_one_even_is_all_odd_or_at_least_two_even_l809_80965

-- Definitions based on the problem conditions
def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

def all_odd (a b c : ℕ) : Prop :=
  ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c

def at_least_two_even (a b c : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c)

-- The proposition to prove
theorem negation_of_one_even_is_all_odd_or_at_least_two_even (a b c : ℕ) :
  ¬ exactly_one_even a b c ↔ all_odd a b c ∨ at_least_two_even a b c :=
by sorry

end negation_of_one_even_is_all_odd_or_at_least_two_even_l809_80965


namespace mutually_exclusive_pairs_l809_80969

-- Define the events based on the conditions
def event_two_red_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 2 ∧ drawn.count "white" = 1)

def event_one_red_two_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 1 ∧ drawn.count "white" = 2)

def event_three_red (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "red" = 3

def event_at_least_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ 1 ≤ drawn.count "white"

def event_three_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "white" = 3

-- Define mutually exclusive property
def mutually_exclusive (A B : List String → List String → Prop) (bag : List String) : Prop :=
  ∀ drawn, A bag drawn → ¬ B bag drawn

-- Define the main theorem statement
theorem mutually_exclusive_pairs (bag : List String) (condition : bag = ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"]) :
  mutually_exclusive event_three_red event_at_least_one_white bag ∧
  mutually_exclusive event_three_red event_three_white bag :=
by
  sorry

end mutually_exclusive_pairs_l809_80969


namespace problem_solution_l809_80961

theorem problem_solution (a b c d e : ℤ) (h : (x - 3)^4 = ax^4 + bx^3 + cx^2 + dx + e) :
  b + c + d + e = 15 :=
by
  sorry

end problem_solution_l809_80961


namespace xiao_wang_program_output_l809_80995

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l809_80995


namespace cube_minus_self_divisible_by_10_l809_80921

theorem cube_minus_self_divisible_by_10 (k : ℤ) : 10 ∣ ((5 * k) ^ 3 - 5 * k) :=
by sorry

end cube_minus_self_divisible_by_10_l809_80921


namespace contractor_fine_per_absent_day_l809_80954

theorem contractor_fine_per_absent_day :
  ∃ x : ℝ, (∀ (total_days absent_days worked_days earnings_per_day total_earnings : ℝ),
   total_days = 30 →
   earnings_per_day = 25 →
   total_earnings = 490 →
   absent_days = 8 →
   worked_days = total_days - absent_days →
   25 * worked_days - absent_days * x = total_earnings
  ) → x = 7.5 :=
by
  existsi 7.5
  intros
  sorry

end contractor_fine_per_absent_day_l809_80954


namespace range_of_a_l809_80943

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, (1 / 2 < x ∧ x < 3) ∧ (x ^ 2 - a * x + 1 = 0)) ↔ (2 ≤ a ∧ a < 10 / 3) :=
by
  sorry

end range_of_a_l809_80943


namespace jason_fires_weapon_every_15_seconds_l809_80917

theorem jason_fires_weapon_every_15_seconds
    (flame_duration_per_fire : ℕ)
    (total_flame_duration_per_minute : ℕ)
    (seconds_per_minute : ℕ)
    (h1 : flame_duration_per_fire = 5)
    (h2 : total_flame_duration_per_minute = 20)
    (h3 : seconds_per_minute = 60) :
    seconds_per_minute / (total_flame_duration_per_minute / flame_duration_per_fire) = 15 := 
by
  sorry

end jason_fires_weapon_every_15_seconds_l809_80917


namespace second_candy_cost_l809_80958

theorem second_candy_cost 
  (C : ℝ) 
  (hp := 25 * 8 + 50 * C = 75 * 6) : 
  C = 5 := 
  sorry

end second_candy_cost_l809_80958


namespace monotonic_increasing_interval_of_f_l809_80974

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logb (1/2) (x^2))

theorem monotonic_increasing_interval_of_f : 
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 0 ∧ -1 ≤ x₂ ∧ x₂ < 0 ∧ x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧ 
  (∀ x : ℝ, f x ≥ 0) := sorry

end monotonic_increasing_interval_of_f_l809_80974


namespace annual_production_2010_l809_80927

-- Defining the parameters
variables (a x : ℝ)

-- Define the growth formula
def annual_growth (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate)^years

-- The statement we need to prove
theorem annual_production_2010 :
  annual_growth a x 5 = a * (1 + x) ^ 5 :=
by
  sorry

end annual_production_2010_l809_80927


namespace gravitational_force_on_asteroid_l809_80922

theorem gravitational_force_on_asteroid :
  ∃ (k : ℝ), ∃ (f : ℝ), 
  (∀ (d : ℝ), f = k / d^2) ∧
  (d = 5000 → f = 700) →
  (∃ (f_asteroid : ℝ), f_asteroid = k / 300000^2 ∧ f_asteroid = 7 / 36) :=
sorry

end gravitational_force_on_asteroid_l809_80922


namespace smallest_possible_value_of_M_l809_80993

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h1 : a + b + c + d + e = 3060) 
    (h2 : a + e ≥ 1300) :
    ∃ M : ℕ, M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) ∧ M = 1174 :=
by
  sorry

end smallest_possible_value_of_M_l809_80993


namespace puzzle_pieces_count_l809_80996

variable (border_pieces : ℕ) (trevor_pieces : ℕ) (joe_pieces : ℕ) (missing_pieces : ℕ)

def total_puzzle_pieces (border_pieces trevor_pieces joe_pieces missing_pieces : ℕ) : ℕ :=
  border_pieces + trevor_pieces + joe_pieces + missing_pieces

theorem puzzle_pieces_count :
  border_pieces = 75 → 
  trevor_pieces = 105 → 
  joe_pieces = 3 * trevor_pieces → 
  missing_pieces = 5 → 
  total_puzzle_pieces border_pieces trevor_pieces joe_pieces missing_pieces = 500 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  -- proof step to get total_number_pieces = 75 + 105 + (3 * 105) + 5
  -- hence total_puzzle_pieces = 500
  sorry

end puzzle_pieces_count_l809_80996


namespace committee_selection_correct_l809_80950

def num_ways_to_choose_committee : ℕ :=
  let total_people := 10
  let president_ways := total_people
  let vp_ways := total_people - 1
  let remaining_people := total_people - 2
  let committee_ways := Nat.choose remaining_people 2
  president_ways * vp_ways * committee_ways

theorem committee_selection_correct :
  num_ways_to_choose_committee = 2520 :=
by
  sorry

end committee_selection_correct_l809_80950


namespace solve_for_y_in_terms_of_x_l809_80926

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : 2 * x - 7 * y = 5) : y = (2 * x - 5) / 7 :=
sorry

end solve_for_y_in_terms_of_x_l809_80926


namespace product_of_possible_values_of_x_l809_80986

theorem product_of_possible_values_of_x :
  (∃ x, |x - 7| - 3 = -2) → ∃ y z, |y - 7| - 3 = -2 ∧ |z - 7| - 3 = -2 ∧ y * z = 48 :=
by
  sorry

end product_of_possible_values_of_x_l809_80986


namespace factor_quadratic_l809_80909

theorem factor_quadratic (x : ℝ) : 
  x^2 + 6 * x = 1 → (x + 3)^2 = 10 := 
by
  intro h
  sorry

end factor_quadratic_l809_80909


namespace max_a_value_l809_80923

theorem max_a_value :
  ∀ (a x : ℝ), 
  (x - 1) * x - (a - 2) * (a + 1) ≥ 1 → a ≤ 3 / 2 := sorry

end max_a_value_l809_80923


namespace range_of_a_l809_80941

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → (a * x^2 - 2 * x + 2) > 0) ↔ (a > 1 / 2) :=
by
  sorry

end range_of_a_l809_80941


namespace find_S20_l809_80915

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom points_collinear (A B C O : α) : Collinear ℝ ({A, B, C} : Set α) ∧ O = 0
axiom vector_relationship (A B C O : α) : O = 0 → C = (a 12) • A + (a 9) • B
axiom line_not_through_origin (A B O : α) : ¬Collinear ℝ ({O, A, B} : Set α)

-- Question: To find S 20
theorem find_S20 (A B C O : α) (h_collinear : Collinear ℝ ({A, B, C} : Set α)) 
  (h_vector : O = 0 → C = (a 12) • A + (a 9) • B) 
  (h_origin : O = 0)
  (h_not_through_origin : ¬Collinear ℝ ({O, A, B} : Set α)) : 
  S 20 = 10 := by
  sorry

end find_S20_l809_80915


namespace hall_length_l809_80946

theorem hall_length (L h : ℝ) (width volume : ℝ) 
  (h_width : width = 6) 
  (h_volume : L * width * h = 108) 
  (h_area : 12 * L = 2 * L * h + 12 * h) : 
  L = 6 := 
  sorry

end hall_length_l809_80946


namespace trig_identity_l809_80929

open Real

theorem trig_identity (θ : ℝ) (h : tan θ = 2) :
  ((sin θ + cos θ) * cos (2 * θ)) / sin θ = -9 / 10 :=
sorry

end trig_identity_l809_80929


namespace simplify_fraction_l809_80920

theorem simplify_fraction : (3 : ℚ) / 462 + 17 / 42 = 95 / 231 :=
by sorry

end simplify_fraction_l809_80920


namespace expected_score_of_basketball_player_l809_80937

theorem expected_score_of_basketball_player :
  let p_inside : ℝ := 0.7
  let p_outside : ℝ := 0.4
  let attempts_inside : ℕ := 10
  let attempts_outside : ℕ := 5
  let points_inside : ℕ := 2
  let points_outside : ℕ := 3
  let E_inside : ℝ := attempts_inside * p_inside * points_inside
  let E_outside : ℝ := attempts_outside * p_outside * points_outside
  E_inside + E_outside = 20 :=
by
  sorry

end expected_score_of_basketball_player_l809_80937
