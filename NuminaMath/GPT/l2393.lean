import Mathlib

namespace cone_base_radius_l2393_239304

theorem cone_base_radius (r_paper : ℝ) (n_parts : ℕ) (r_cone_base : ℝ) 
  (h_radius_paper : r_paper = 16)
  (h_n_parts : n_parts = 4)
  (h_cone_part : r_cone_base = r_paper / n_parts) : r_cone_base = 4 := by
  sorry

end cone_base_radius_l2393_239304


namespace ken_paid_20_l2393_239395

section
variable (pound_price : ℤ) (pounds_bought : ℤ) (change_received : ℤ)
variable (total_cost : ℤ) (amount_paid : ℤ)

-- Conditions
def price_per_pound := 7  -- A pound of steak costs $7
def pounds_bought_value := 2  -- Ken bought 2 pounds of steak
def change_received_value := 6  -- Ken received $6 back after paying

-- Intermediate Calculations
def total_cost_of_steak := pounds_bought_value * price_per_pound  -- Total cost of steak
def amount_paid_calculated := total_cost_of_steak + change_received_value  -- Amount paid based on total cost and change received

-- Problem Statement
theorem ken_paid_20 : (total_cost_of_steak = total_cost) ∧ (amount_paid_calculated = amount_paid) -> amount_paid = 20 :=
by
  intros h
  sorry
end

end ken_paid_20_l2393_239395


namespace least_xy_l2393_239386

noncomputable def condition (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 1 / 7)

theorem least_xy (x y : ℕ) (h : condition x y) : x * y = 98 :=
sorry

end least_xy_l2393_239386


namespace sum_of_sequence_eq_six_seventeenth_l2393_239398

noncomputable def cn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.cos (n * Real.arctan (2 / 3))
noncomputable def dn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.sin (n * Real.arctan (2 / 3))

theorem sum_of_sequence_eq_six_seventeenth : 
  (∑' n : ℕ, (cn n * dn n / 8^n)) = 6/17 := sorry

end sum_of_sequence_eq_six_seventeenth_l2393_239398


namespace exist_c_l2393_239372

theorem exist_c (p : ℕ) (r : ℤ) (a b : ℤ) [Fact (Nat.Prime p)]
  (hp1 : r^7 ≡ 1 [ZMOD p])
  (hp2 : r + 1 - a^2 ≡ 0 [ZMOD p])
  (hp3 : r^2 + 1 - b^2 ≡ 0 [ZMOD p]) :
  ∃ c : ℤ, (r^3 + 1 - c^2) ≡ 0 [ZMOD p] :=
by
  sorry

end exist_c_l2393_239372


namespace pinecones_left_l2393_239358

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end pinecones_left_l2393_239358


namespace train_crosses_platform_in_20_seconds_l2393_239355

theorem train_crosses_platform_in_20_seconds 
  (t : ℝ) (lp : ℝ) (lt : ℝ) (tp : ℝ) (sp : ℝ) (st : ℝ) 
  (pass_time : st = lt / tp) (lc : lp = 267) (lc_train : lt = 178) (cross_time : t = sp / st) : 
  t = 20 :=
by
  sorry

end train_crosses_platform_in_20_seconds_l2393_239355


namespace problem1_problem2_l2393_239382

variable (a b : ℝ)

theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  1/a + 1/(b+1) ≥ 4/5 := by
  sorry

theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  4/(a*b) + a/b ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end problem1_problem2_l2393_239382


namespace only_point_D_lies_on_graph_l2393_239359

def point := ℤ × ℤ

def lies_on_graph (f : ℤ → ℤ) (p : point) : Prop :=
  f p.1 = p.2

def f (x : ℤ) : ℤ := 2 * x - 1

theorem only_point_D_lies_on_graph :
  (lies_on_graph f (-1, 3) = false) ∧ 
  (lies_on_graph f (0, 1) = false) ∧ 
  (lies_on_graph f (1, -1) = false) ∧ 
  (lies_on_graph f (2, 3)) := 
by
  sorry

end only_point_D_lies_on_graph_l2393_239359


namespace simplify_expression_l2393_239363

theorem simplify_expression (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 :=
sorry

end simplify_expression_l2393_239363


namespace intersection_subset_complement_l2393_239375

open Set

variable (U A B : Set ℕ)

theorem intersection_subset_complement (U : Set ℕ) (A B : Set ℕ) 
  (hU: U = {1, 2, 3, 4, 5, 6}) 
  (hA: A = {1, 3, 5}) 
  (hB: B = {2, 4, 5}) : 
  A ∩ (U \ B) = {1, 3} := 
by
  sorry

end intersection_subset_complement_l2393_239375


namespace find_value_of_A_l2393_239313

theorem find_value_of_A (M T A E : ℕ) (H : ℕ := 8) 
  (h1 : M + A + T + H = 28) 
  (h2 : T + E + A + M = 34) 
  (h3 : M + E + E + T = 30) : 
  A = 16 :=
by 
  sorry

end find_value_of_A_l2393_239313


namespace david_reading_time_l2393_239351

theorem david_reading_time (total_time : ℕ) (math_time : ℕ) (spelling_time : ℕ) 
  (reading_time : ℕ) (h1 : total_time = 60) (h2 : math_time = 15) 
  (h3 : spelling_time = 18) (h4 : reading_time = total_time - (math_time + spelling_time)) : 
  reading_time = 27 := by
  sorry

end david_reading_time_l2393_239351


namespace greatest_value_of_x_is_20_l2393_239390

noncomputable def greatest_multiple_of_4 (x : ℕ) : Prop :=
  (x % 4 = 0 ∧ x^2 < 500 ∧ ∀ y : ℕ, (y % 4 = 0 ∧ y^2 < 500) → y ≤ x)

theorem greatest_value_of_x_is_20 : greatest_multiple_of_4 20 :=
  by 
  sorry

end greatest_value_of_x_is_20_l2393_239390


namespace mistaken_multiplication_l2393_239305

theorem mistaken_multiplication (x : ℕ) : 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  (a * b - a * x = incorrect_result) ↔ (x = 34) := 
by 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  sorry

end mistaken_multiplication_l2393_239305


namespace relationship_m_n_l2393_239373

variables {a b : ℝ}

theorem relationship_m_n (h1 : |a| ≠ |b|) (m : ℝ) (n : ℝ)
  (hm : m = (|a| - |b|) / |a - b|)
  (hn : n = (|a| + |b|) / |a + b|) :
  m ≤ n :=
by sorry

end relationship_m_n_l2393_239373


namespace pencils_per_child_l2393_239397

-- Define the conditions
def totalPencils : ℕ := 18
def numberOfChildren : ℕ := 9

-- The proof problem
theorem pencils_per_child : totalPencils / numberOfChildren = 2 := 
by
  sorry

end pencils_per_child_l2393_239397


namespace monochromatic_regions_lower_bound_l2393_239318

theorem monochromatic_regions_lower_bound (n : ℕ) (h_n_ge_2 : n ≥ 2) :
  ∀ (blue_lines red_lines : ℕ) (conditions :
    blue_lines = 2 * n ∧ red_lines = n ∧ 
    (∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      (blue_lines = 2 * n ∧ red_lines = n))) 
  , ∃ (monochromatic_regions : ℕ), 
      monochromatic_regions ≥ (n - 1) * (n - 2) / 2 :=
sorry

end monochromatic_regions_lower_bound_l2393_239318


namespace length_of_FD_l2393_239340

/-- Square ABCD with side length 8 cm, corner C is folded to point E on AD such that AE = 2 cm and ED = 6 cm. Find the length of FD. -/
theorem length_of_FD 
  (A B C D E F G : Type)
  (square_length : Float)
  (AD_length AE_length ED_length : Float)
  (hyp1 : square_length = 8)
  (hyp2 : AE_length = 2)
  (hyp3 : ED_length = 6)
  (hyp4 : AD_length = AE_length + ED_length)
  (FD_length : Float) :
  FD_length = 7 / 4 := 
  by 
  sorry

end length_of_FD_l2393_239340


namespace set_union_example_l2393_239331

variable (A B : Set ℝ)

theorem set_union_example :
  A = {x | -2 < x ∧ x ≤ 1} ∧ B = {x | -1 ≤ x ∧ x < 2} →
  (A ∪ B) = {x | -2 < x ∧ x < 2} := 
by
  sorry

end set_union_example_l2393_239331


namespace min_u_condition_l2393_239337

-- Define the function u and the condition
def u (x y : ℝ) : ℝ := x^2 + 4 * x + y^2 - 2 * y

def condition (x y : ℝ) : Prop := 2 * x + y ≥ 1

-- The statement we want to prove
theorem min_u_condition : ∃ (x y : ℝ), condition x y ∧ u x y = -9/5 := 
by
  sorry

end min_u_condition_l2393_239337


namespace solve_for_s_l2393_239307

theorem solve_for_s {x : ℝ} (h : 4 * x^2 - 8 * x - 320 = 0) : ∃ s, s = 81 :=
by 
  -- Introduce the conditions and the steps
  sorry

end solve_for_s_l2393_239307


namespace tangent_slope_of_cubic_l2393_239371

theorem tangent_slope_of_cubic (P : ℝ × ℝ) (tangent_at_P : ℝ) (h1 : P.snd = P.fst ^ 3)
  (h2 : tangent_at_P = 3) : P = (1,1) ∨ P = (-1,-1) :=
by
  sorry

end tangent_slope_of_cubic_l2393_239371


namespace find_number_l2393_239330

theorem find_number (x : ℝ) (h : 0.65 * x = 0.05 * 60 + 23) : x = 40 :=
sorry

end find_number_l2393_239330


namespace donut_combinations_l2393_239324

theorem donut_combinations (donuts types : ℕ) (at_least_one : ℕ) :
  donuts = 7 ∧ types = 5 ∧ at_least_one = 4 → ∃ combinations : ℕ, combinations = 100 :=
by
  intros h
  sorry

end donut_combinations_l2393_239324


namespace overlapping_area_of_thirty_sixty_ninety_triangles_l2393_239321

-- Definitions for 30-60-90 triangle and the overlapping region
def thirty_sixty_ninety_triangle (hypotenuse : ℝ) := 
  (hypotenuse > 0) ∧ 
  (exists (short_leg long_leg : ℝ), short_leg = hypotenuse / 2 ∧ long_leg = short_leg * (Real.sqrt 3))

-- Area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ :=
  base * height

theorem overlapping_area_of_thirty_sixty_ninety_triangles :
  ∀ (hypotenuse : ℝ), thirty_sixty_ninety_triangle hypotenuse →
  hypotenuse = 10 →
  (∃ (base height : ℝ), base = height ∧ base * height = parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3)) →
  parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3) = 75 :=
by
  sorry

end overlapping_area_of_thirty_sixty_ninety_triangles_l2393_239321


namespace minimize_cost_l2393_239396

-- Define the unit prices of the soccer balls.
def price_A := 50
def price_B := 80

-- Define the condition for the total number of balls and cost function.
def total_balls := 80
def cost (a : ℕ) : ℕ := price_A * a + price_B * (total_balls - a)
def valid_a (a : ℕ) : Prop := 30 ≤ a ∧ a ≤ (3 * (total_balls - a))

-- Prove the number of brand A soccer balls to minimize the total cost.
theorem minimize_cost : ∃ a : ℕ, valid_a a ∧ ∀ b : ℕ, valid_a b → cost a ≤ cost b :=
sorry

end minimize_cost_l2393_239396


namespace part1_part2_l2393_239310

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x)

theorem part1 : f (Real.pi / 8) = Real.sqrt 2 + 1 := sorry

theorem part2 : (∀ x1 x2 : ℝ, f (x1 + Real.pi) = f x1) ∧ (∀ x : ℝ, f x ≥ 1 - Real.sqrt 2) := 
  sorry

-- Explanation:
-- part1 is for proving f(π/8) = √2 + 1
-- part2 handles proving the smallest positive period and the minimum value of the function.

end part1_part2_l2393_239310


namespace taxi_ride_cost_l2393_239342

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l2393_239342


namespace mod_37_5_l2393_239326

theorem mod_37_5 : 37 % 5 = 2 := 
by
  sorry

end mod_37_5_l2393_239326


namespace perpendicular_vector_l2393_239391

-- Vectors a and b are given
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

-- Defining the vector addition and scalar multiplication for our context
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (m : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (m * v.1, m * v.2)

-- The vector a + m * b
def a_plus_m_b (m : ℝ) : ℝ × ℝ := vector_add a (scalar_mul m b)

-- The dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The statement that a is perpendicular to (a + m * b) when m = 5
theorem perpendicular_vector : dot_product a (a_plus_m_b 5) = 0 :=
sorry

end perpendicular_vector_l2393_239391


namespace gcd_of_polynomial_l2393_239399

theorem gcd_of_polynomial (a : ℤ) (h : 720 ∣ a) : Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := 
by 
  sorry

end gcd_of_polynomial_l2393_239399


namespace center_of_circle_l2393_239380

theorem center_of_circle : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → (1, 1) = (1, 1) :=
by
  intros x y h
  sorry

end center_of_circle_l2393_239380


namespace difference_of_place_values_l2393_239357

theorem difference_of_place_values :
  let n := 54179759
  let pos1 := 10000 * 7
  let pos2 := 10 * 7
  pos1 - pos2 = 69930 := by
  sorry

end difference_of_place_values_l2393_239357


namespace complete_square_q_value_l2393_239356

theorem complete_square_q_value :
  ∃ p q, (16 * x^2 - 32 * x - 512 = 0) ∧ ((x + p)^2 = q) → q = 33 := by
  sorry

end complete_square_q_value_l2393_239356


namespace solve_equation_l2393_239341

theorem solve_equation {x : ℝ} (hx : x = 1) : 9 - 3 / x / 3 + 3 = 3 := by
  rw [hx] -- Substitute x = 1
  norm_num -- Simplify the numerical expression
  sorry -- to be proved

end solve_equation_l2393_239341


namespace amount_of_flour_already_put_in_l2393_239387

theorem amount_of_flour_already_put_in 
  (total_flour_needed : ℕ) (flour_remaining : ℕ) (x : ℕ) 
  (h1 : total_flour_needed = 9) 
  (h2 : flour_remaining = 7) 
  (h3 : total_flour_needed - flour_remaining = x) : 
  x = 2 := 
sorry

end amount_of_flour_already_put_in_l2393_239387


namespace ellipse_area_l2393_239385

-- Definitions based on the conditions
def cylinder_height : ℝ := 10
def cylinder_base_radius : ℝ := 1

-- Equivalent Proof Problem Statement
theorem ellipse_area
  (h : ℝ := cylinder_height)
  (r : ℝ := cylinder_base_radius)
  (ball_position_lower : ℝ := -4) -- derived from - (h / 2 - r)
  (ball_position_upper : ℝ := 4) -- derived from  (h / 2 - r)
  : (π * 4 * 2 = 16 * π) :=
by
  sorry

end ellipse_area_l2393_239385


namespace sliderB_moves_distance_l2393_239335

theorem sliderB_moves_distance :
  ∀ (A B : ℝ) (rod_length : ℝ),
    (A = 20) →
    (B = 15) →
    (rod_length = Real.sqrt (20^2 + 15^2)) →
    (rod_length = 25) →
    (B_new = 25 - 15) →
    B_new = 10 := by
  sorry

end sliderB_moves_distance_l2393_239335


namespace initial_blue_balls_l2393_239315

theorem initial_blue_balls (total_balls : ℕ) (remaining_balls : ℕ) (B : ℕ) :
  total_balls = 18 → remaining_balls = total_balls - 3 → (B - 3) / remaining_balls = 1 / 5 → B = 6 :=
by 
  intros htotal hremaining hprob
  sorry

end initial_blue_balls_l2393_239315


namespace no_integral_roots_l2393_239379

theorem no_integral_roots :
  ¬(∃ (x : ℤ), 5 * x^2 + 3 = 40) ∧
  ¬(∃ (x : ℤ), (3 * x - 2)^3 = (x - 2)^3 - 27) ∧
  ¬(∃ (x : ℤ), x^2 - 4 = 3 * x - 4) :=
by sorry

end no_integral_roots_l2393_239379


namespace problem_1_problem_2_l2393_239343

theorem problem_1 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
sorry

theorem problem_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_2 + a_4 + a_6 = 365 :=
sorry

end problem_1_problem_2_l2393_239343


namespace product_of_real_roots_l2393_239303

theorem product_of_real_roots : 
  let f (x : ℝ) := x ^ Real.log x / Real.log 2 
  ∃ r1 r2 : ℝ, (f r1 = 16 ∧ f r2 = 16) ∧ (r1 * r2 = 1) := 
by
  sorry

end product_of_real_roots_l2393_239303


namespace n19_minus_n7_div_30_l2393_239328

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l2393_239328


namespace smallest_positive_integer_n_l2393_239345

noncomputable def matrix_330 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (330 * Real.pi / 180), -Real.sin (330 * Real.pi / 180)],
    ![Real.sin (330 * Real.pi / 180), Real.cos (330 * Real.pi / 180)]
  ]

theorem smallest_positive_integer_n (n : ℕ) (h : matrix_330 ^ n = 1) : n = 12 := sorry

end smallest_positive_integer_n_l2393_239345


namespace parallel_lines_slope_l2393_239327

theorem parallel_lines_slope {a : ℝ} (h : -a / 3 = -2 / 3) : a = 2 := 
by
  sorry

end parallel_lines_slope_l2393_239327


namespace solve_recursive_fraction_l2393_239325

noncomputable def recursive_fraction (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => 1 + 1 / (recursive_fraction n x)

theorem solve_recursive_fraction (x : ℝ) (n : ℕ) :
  (recursive_fraction n x = x) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) :=
sorry

end solve_recursive_fraction_l2393_239325


namespace find_even_integer_l2393_239308

theorem find_even_integer (x y z : ℤ) (h₁ : Even x) (h₂ : Odd y) (h₃ : Odd z)
  (h₄ : x < y) (h₅ : y < z) (h₆ : y - x > 5) (h₇ : z - x = 9) : x = 2 := 
by 
  sorry

end find_even_integer_l2393_239308


namespace seating_arrangement_l2393_239300

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem seating_arrangement : 
  let republicans := 6
  let democrats := 4
  (factorial (republicans - 1)) * (binom republicans democrats) * (factorial democrats) = 43200 :=
by
  sorry

end seating_arrangement_l2393_239300


namespace total_students_in_class_l2393_239361

-- No need for noncomputable def here as we're dealing with basic arithmetic

theorem total_students_in_class (jellybeans_total jellybeans_left boys_girls_diff : ℕ)
  (girls boys students : ℕ) :
  jellybeans_total = 450 →
  jellybeans_left = 10 →
  boys_girls_diff = 3 →
  boys = girls + boys_girls_diff →
  students = girls + boys →
  (girls * girls) + (boys * boys) = jellybeans_total - jellybeans_left →
  students = 29 := 
by
  intro h_total h_left h_diff h_boys h_students h_distribution
  sorry

end total_students_in_class_l2393_239361


namespace proper_subsets_count_l2393_239360

theorem proper_subsets_count (A : Set (Fin 4)) (h : A = {1, 2, 3}) : 
  ∃ n : ℕ, n = 7 ∧ ∃ (S : Finset (Set (Fin 4))), S.card = n ∧ (∀ B, B ∈ S → B ⊂ A) := 
by {
  sorry
}

end proper_subsets_count_l2393_239360


namespace distinct_students_count_l2393_239384

theorem distinct_students_count
  (algebra_students : ℕ)
  (calculus_students : ℕ)
  (statistics_students : ℕ)
  (algebra_statistics_overlap : ℕ)
  (no_other_overlaps : algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32) :
  algebra_students = 13 → calculus_students = 10 → statistics_students = 12 → algebra_statistics_overlap = 3 → 
  algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end distinct_students_count_l2393_239384


namespace sum_first_70_odd_eq_4900_l2393_239350

theorem sum_first_70_odd_eq_4900 (h : (70 * (70 + 1) = 4970)) :
  (70 * 70 = 4900) :=
by
  sorry

end sum_first_70_odd_eq_4900_l2393_239350


namespace scoring_situations_4_students_l2393_239302

noncomputable def number_of_scoring_situations (students : ℕ) (topicA_score : ℤ) (topicB_score : ℤ) : ℕ :=
  let combinations := Nat.choose 4 2
  let first_category := combinations * 2 * 2
  let second_category := 2 * combinations
  first_category + second_category

theorem scoring_situations_4_students : number_of_scoring_situations 4 100 90 = 36 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end scoring_situations_4_students_l2393_239302


namespace set_inter_compl_eq_l2393_239306

def U := ℝ
def M : Set ℝ := { x | abs (x - 1/2) ≤ 5/2 }
def P : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def complement_U_M : Set ℝ := { x | x < -2 ∨ x > 3 }

theorem set_inter_compl_eq :
  (complement_U_M ∩ P) = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end set_inter_compl_eq_l2393_239306


namespace monthly_expenses_last_month_l2393_239370

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l2393_239370


namespace max_value_of_trig_expr_l2393_239309

theorem max_value_of_trig_expr (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 :=
sorry

end max_value_of_trig_expr_l2393_239309


namespace find_n_l2393_239365

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n_l2393_239365


namespace scientific_notation_of_2135_billion_l2393_239344

theorem scientific_notation_of_2135_billion :
  (2135 * 10^9 : ℝ) = 2.135 * 10^11 := by
  sorry

end scientific_notation_of_2135_billion_l2393_239344


namespace sequences_get_arbitrarily_close_l2393_239301

noncomputable def a_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^n
noncomputable def b_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^(n + 1)

theorem sequences_get_arbitrarily_close (n : ℕ) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b_n n - a_n n| < ε :=
sorry

end sequences_get_arbitrarily_close_l2393_239301


namespace calculate_expression_l2393_239319

theorem calculate_expression : 3 * Real.sqrt 2 - abs (Real.sqrt 2 - Real.sqrt 3) = 4 * Real.sqrt 2 - Real.sqrt 3 :=
  by sorry

end calculate_expression_l2393_239319


namespace replacement_parts_l2393_239311

theorem replacement_parts (num_machines : ℕ) (parts_per_machine : ℕ) (week1_fail_rate : ℚ) (week2_fail_rate : ℚ) (week3_fail_rate : ℚ) :
  num_machines = 500 ->
  parts_per_machine = 6 ->
  week1_fail_rate = 0.10 ->
  week2_fail_rate = 0.30 ->
  week3_fail_rate = 0.60 ->
  (num_machines * parts_per_machine) * week1_fail_rate +
  (num_machines * parts_per_machine) * week2_fail_rate +
  (num_machines * parts_per_machine) * week3_fail_rate = 3000 := by
  sorry

end replacement_parts_l2393_239311


namespace neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l2393_239381

-- (1) Prove -(-2) = 2
theorem neg_neg_two : -(-2) = 2 := 
sorry

-- (2) Prove -6 + 6 = 0
theorem neg_six_plus_six : -6 + 6 = 0 := 
sorry

-- (3) Prove (-3) * 5 = -15
theorem neg_three_times_five : (-3) * 5 = -15 := 
sorry

-- (4) Prove 2x - 3x = -x
theorem two_x_minus_three_x (x : ℝ) : 2 * x - 3 * x = - x := 
sorry

end neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l2393_239381


namespace smallest_x_l2393_239354

theorem smallest_x (x : ℤ) (h : x + 3 < 3 * x - 4) : x = 4 :=
by
  sorry

end smallest_x_l2393_239354


namespace price_after_discount_l2393_239392

-- Define the original price and discount
def original_price : ℕ := 76
def discount : ℕ := 25

-- The main proof statement
theorem price_after_discount : original_price - discount = 51 := by
  sorry

end price_after_discount_l2393_239392


namespace pow_log_sqrt_l2393_239367

theorem pow_log_sqrt (a b c : ℝ) (h1 : a = 81) (h2 : b = 500) (h3 : c = 3) :
  ((a ^ (Real.log b / Real.log c)) ^ (1 / 2)) = 250000 :=
by
  sorry

end pow_log_sqrt_l2393_239367


namespace minimum_value_f_inequality_proof_l2393_239312

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 1)

-- The minimal value of f(x)
def m : ℝ := 4

theorem minimum_value_f :
  (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, -3 ≤ x ∧ x ≤ 1 ∧ f x = m) :=
by
  sorry -- Proof that the minimum value of f(x) is 4 and occurs in the range -3 ≤ x ≤ 1

variables (p q r : ℝ)

-- Given condition that p^2 + 2q^2 + r^2 = 4
theorem inequality_proof (h : p^2 + 2 * q^2 + r^2 = m) : q * (p + r) ≤ 2 :=
by
  sorry -- Proof that q(p + r) ≤ 2 given p^2 + 2q^2 + r^2 = 4

end minimum_value_f_inequality_proof_l2393_239312


namespace scientific_notation_of_935000000_l2393_239368

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l2393_239368


namespace probability_of_picking_peach_l2393_239317

-- Define the counts of each type of fruit
def apples : ℕ := 5
def pears : ℕ := 3
def peaches : ℕ := 2

-- Define the total number of fruits
def total_fruits : ℕ := apples + pears + peaches

-- Define the probability of picking a peach
def probability_of_peach : ℚ := peaches / total_fruits

-- State the theorem
theorem probability_of_picking_peach : probability_of_peach = 1/5 := by
  -- proof goes here
  sorry

end probability_of_picking_peach_l2393_239317


namespace problem_statement_eq_l2393_239349

noncomputable def given_sequence (a : ℝ) (n : ℕ) : ℝ :=
  a^n

noncomputable def Sn (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  (a / (a - 1)) * (an - 1)

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  2 * (Sn a n (given_sequence a n)) / (given_sequence a n) + 1

noncomputable def cn (a : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (bn a n)

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc k => acc + cn a (k + 1)) 0

theorem problem_statement_eq :
  ∀ (a : ℝ) (n : ℕ), a ≠ 0 → a ≠ 1 →
  (bn a n = (3:ℝ)^n) →
  Tn (1 / 3) n = 3^(n+1) * (2 * n - 3) / 4 + 9 / 4 :=
by
  intros
  sorry

end problem_statement_eq_l2393_239349


namespace mod_product_l2393_239364

theorem mod_product :
  (105 * 86 * 97) % 25 = 10 :=
by
  sorry

end mod_product_l2393_239364


namespace tangent_line_at_A_l2393_239347

def f (x : ℝ) : ℝ := x ^ (1 / 2)

def tangent_line_equation (x y: ℝ) : Prop :=
  4 * x - 4 * y + 1 = 0

theorem tangent_line_at_A :
  tangent_line_equation (1/4) (f (1/4)) :=
by
  sorry

end tangent_line_at_A_l2393_239347


namespace find_x0_l2393_239393

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 1
else if x < 0 then -x^2 + 1
else 0

theorem find_x0 :
  ∃ x0 : ℝ, f x0 = 1/2 ∧ x0 = -Real.sqrt 2 / 2 :=
by
  sorry

end find_x0_l2393_239393


namespace simplify_logical_expression_l2393_239338

variables (A B C : Bool)

theorem simplify_logical_expression :
  (A && !B || B && !C || B && C || A && B) = (A || B) :=
by { sorry }

end simplify_logical_expression_l2393_239338


namespace chef_initial_potatoes_l2393_239383

theorem chef_initial_potatoes (fries_per_potato : ℕ) (total_fries_needed : ℕ) (leftover_potatoes : ℕ) 
  (H1 : fries_per_potato = 25) 
  (H2 : total_fries_needed = 200) 
  (H3 : leftover_potatoes = 7) : 
  (total_fries_needed / fries_per_potato + leftover_potatoes = 15) :=
by
  sorry

end chef_initial_potatoes_l2393_239383


namespace simplify_and_evaluate_expression_l2393_239329

theorem simplify_and_evaluate_expression (m : ℕ) (h : m = 2) :
  ( (↑m + 1) / (↑m - 1) + 1 ) / ( (↑m + m^2) / (m^2 - 2*m + 1) ) - ( 2 - 2*↑m ) / ( m^2 - 1 ) = 4 / 3 :=
by sorry

end simplify_and_evaluate_expression_l2393_239329


namespace domain_of_f_l2393_239336

theorem domain_of_f (x : ℝ) : (2*x - x^2 > 0 ∧ x ≠ 1) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) :=
by
  -- proof omitted
  sorry

end domain_of_f_l2393_239336


namespace size_relationship_l2393_239322

variable (a1 a2 b1 b2 : ℝ)

theorem size_relationship (h1 : a1 < a2) (h2 : b1 < b2) : a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end size_relationship_l2393_239322


namespace record_jump_l2393_239339

theorem record_jump (standard_jump jump : Float) (h_standard : standard_jump = 4.00) (h_jump : jump = 3.85) : (jump - standard_jump : Float) = -0.15 := 
by
  rw [h_standard, h_jump]
  simp
  sorry

end record_jump_l2393_239339


namespace triangle_median_difference_l2393_239316

theorem triangle_median_difference
    (A B C D E : Type)
    (BC_len : BC = 10)
    (AD_len : AD = 6)
    (BE_len : BE = 7.5) :
    ∃ X_max X_min : ℝ, 
    X_max = AB^2 + AC^2 + BC^2 ∧ 
    X_min = AB^2 + AC^2 + BC^2 ∧ 
    (X_max - X_min) = 56.25 :=
by
  sorry

end triangle_median_difference_l2393_239316


namespace posters_count_l2393_239376

-- Define the regular price per poster
def regular_price : ℕ := 4

-- Jeremy can buy 24 posters at regular price
def posters_at_regular_price : ℕ := 24

-- Total money Jeremy has is equal to the money needed to buy 24 posters
def total_money : ℕ := posters_at_regular_price * regular_price

-- The special deal: buy one get the second at half price
def cost_of_two_posters : ℕ := regular_price + regular_price / 2

-- Number of pairs Jeremy can buy with his total money
def number_of_pairs : ℕ := total_money / cost_of_two_posters

-- Total number of posters Jeremy can buy under the sale
def total_posters := number_of_pairs * 2

-- Prove that the total posters is 32
theorem posters_count : total_posters = 32 := by
  sorry

end posters_count_l2393_239376


namespace median_length_of_pieces_is_198_l2393_239394

   -- Define the conditions
   variables (A B C D E : ℕ)
   variables (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E)
   variables (avg_length : (A + B + C + D + E) = 640)
   variables (h_A_max : A ≤ 110)

   -- Statement of the problem (proof stub)
   theorem median_length_of_pieces_is_198 :
     C = 198 :=
   by
   sorry
   
end median_length_of_pieces_is_198_l2393_239394


namespace find_k_l2393_239333

noncomputable def vector_a : ℝ × ℝ := (-1, 1)
noncomputable def vector_b : ℝ × ℝ := (2, 3)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (-2, k)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) (h : perp (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) (vector_c k)) : k = 1 / 2 :=
by
  sorry

end find_k_l2393_239333


namespace extremum_of_f_l2393_239389

def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

theorem extremum_of_f :
  ∃ (xmin xmax : ℝ) (x1 y1 : ℝ), f x1 y1 = xmin ∧ (x1 = 11 ∧ y1 = 2) ∧
  ∃ (xmax : ℝ) (x2 y2 : ℝ), f x2 y2 = xmax ∧ (x2 = 1 ∧ y2 = 4) ∧
  xmin = 10 ∧ xmax = 570 := 
by
  sorry

end extremum_of_f_l2393_239389


namespace remainder_of_power_is_41_l2393_239377

theorem remainder_of_power_is_41 : 
  ∀ (n k : ℕ), n = 2019 → k = 2018 → (n^k) % 100 = 41 :=
  by 
    intros n k hn hk 
    rw [hn, hk] 
    exact sorry

end remainder_of_power_is_41_l2393_239377


namespace actual_distance_between_mountains_l2393_239314

theorem actual_distance_between_mountains (D_map : ℝ) (d_map_ram : ℝ) (d_real_ram : ℝ)
  (hD_map : D_map = 312) (hd_map_ram : d_map_ram = 25) (hd_real_ram : d_real_ram = 10.897435897435898) :
  D_map / d_map_ram * d_real_ram = 136 :=
by
  -- Theorem statement is proven based on the given conditions.
  sorry

end actual_distance_between_mountains_l2393_239314


namespace determine_a_from_root_l2393_239348

noncomputable def quadratic_eq (x a : ℝ) : Prop := x^2 - a = 0

theorem determine_a_from_root :
  (∃ a : ℝ, quadratic_eq 2 a) → (∃ a : ℝ, a = 4) :=
by
  intro h
  obtain ⟨a, ha⟩ := h
  use a
  have h_eq : 2^2 - a = 0 := ha
  linarith

end determine_a_from_root_l2393_239348


namespace no_integer_b_two_distinct_roots_l2393_239362

theorem no_integer_b_two_distinct_roots :
  ∀ b : ℤ, ¬ ∃ x y : ℤ, x ≠ y ∧ (x^4 + 4 * x^3 + b * x^2 + 16 * x + 8 = 0) ∧ (y^4 + 4 * y^3 + b * y^2 + 16 * y + 8 = 0) :=
by
  sorry

end no_integer_b_two_distinct_roots_l2393_239362


namespace candy_mixture_l2393_239369

theorem candy_mixture (x : ℝ) (h1 : x * 3 + 64 * 2 = (x + 64) * 2.2) : x + 64 = 80 :=
by sorry

end candy_mixture_l2393_239369


namespace mark_total_theater_spending_l2393_239378

def week1_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week2_cost : ℝ := (2.5 * 6 - 0.1 * (2.5 * 6)) + 3
def week3_cost : ℝ := 4 * 4 + 3
def week4_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week5_cost : ℝ := (2 * (3.5 * 6 - 0.1 * (3.5 * 6))) + 6
def week6_cost : ℝ := 2 * 7 + 3

def total_cost : ℝ := week1_cost + week2_cost + week3_cost + week4_cost + week5_cost + week6_cost

theorem mark_total_theater_spending : total_cost = 126.30 := sorry

end mark_total_theater_spending_l2393_239378


namespace chord_length_intercepted_by_line_on_circle_l2393_239388

theorem chord_length_intercepted_by_line_on_circle :
  ∀ (ρ θ : ℝ), (ρ = 4) →
  (ρ * Real.sin (θ + (Real.pi / 4)) = 2) →
  (4 * Real.sqrt (16 - (2 ^ 2)) = 4 * Real.sqrt 3) :=
by
  intros ρ θ hρ hline_eq
  sorry

end chord_length_intercepted_by_line_on_circle_l2393_239388


namespace evaluate_expression_at_neg3_l2393_239374

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l2393_239374


namespace arithmetic_sequence_a2_a4_a9_eq_18_l2393_239352

theorem arithmetic_sequence_a2_a4_a9_eq_18 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : S 9 = 54) 
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 2 + a 4 + a 9 = 18 :=
sorry

end arithmetic_sequence_a2_a4_a9_eq_18_l2393_239352


namespace rectangle_length_l2393_239323

theorem rectangle_length (w l : ℝ) (hP : (2 * l + 2 * w) / w = 5) (hA : l * w = 150) : l = 15 :=
by
  sorry

end rectangle_length_l2393_239323


namespace floor_sqrt_120_eq_10_l2393_239334

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l2393_239334


namespace find_bk_l2393_239320

theorem find_bk
  (A B C D : ℝ)
  (BC : ℝ) (hBC : BC = 3)
  (AB CD : ℝ) (hAB_CD : AB = 2 * CD)
  (BK : ℝ) (hBK : BK = 2) :
  ∃ x a : ℝ, (x = BK) ∧ (AB = 2 * CD) ∧ ((2 * a + x) * (3 - x) = x * (a + 3 - x)) :=
by
  sorry

end find_bk_l2393_239320


namespace interval_of_increase_l2393_239366

noncomputable def u (x : ℝ) : ℝ := x^2 - 5*x + 6

def increasing_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := Real.log (u x)

theorem interval_of_increase :
  increasing_interval f {x : ℝ | 3 < x} :=
sorry

end interval_of_increase_l2393_239366


namespace find_a_find_k_max_l2393_239346

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a_find_k_max_l2393_239346


namespace eventually_periodic_l2393_239332

variable (u : ℕ → ℤ)

def bounded (u : ℕ → ℤ) : Prop :=
  ∃ (m M : ℤ), ∀ (n : ℕ), m ≤ u n ∧ u n ≤ M

def recurrence (u : ℕ → ℤ) (n : ℕ) : Prop := 
  u (n) = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

theorem eventually_periodic (hu_bounded : bounded u) (hu_recurrence : ∀ n ≥ 4, recurrence u n) :
  ∃ N M, ∀ k ≥ 0, u (N + k) = u (N + M + k) :=
sorry

end eventually_periodic_l2393_239332


namespace solve_equations_l2393_239353

theorem solve_equations :
  (∀ x, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x, x^2 - 6 * x + 9 = 0 ↔ x = 3) ∧
  (∀ x, x^2 - 7 * x + 12 = 0 ↔ x = 3 ∨ x = 4) ∧
  (∀ x, 2 * x^2 - 3 * x - 5 = 0 ↔ x = 5 / 2 ∨ x = -1) :=
by
  -- Proof goes here
  sorry

end solve_equations_l2393_239353
