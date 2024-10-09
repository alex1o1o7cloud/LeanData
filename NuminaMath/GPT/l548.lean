import Mathlib

namespace equations_of_line_l548_54884

variables (x y : ℝ)

-- Given conditions
def passes_through_point (P : ℝ × ℝ) (x y : ℝ) := (x, y) = P

def has_equal_intercepts_on_axes (f : ℝ → ℝ) :=
  ∃ z : ℝ, z ≠ 0 ∧ f z = 0 ∧ f 0 = z

-- The proof problem statement
theorem equations_of_line (P : ℝ × ℝ) (hP : passes_through_point P 2 (-3)) (h : has_equal_intercepts_on_axes (λ x => -x / (x / 2))) :
  (x + y + 1 = 0) ∨ (3 * x + 2 * y = 0) := 
sorry

end equations_of_line_l548_54884


namespace paco_min_cookies_l548_54841

theorem paco_min_cookies (x : ℕ) (h_initial : 25 - x ≥ 0) : 
  x + (3 + 2) ≥ 5 := by
  sorry

end paco_min_cookies_l548_54841


namespace g_of_1_equals_3_l548_54866

theorem g_of_1_equals_3 (f g : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_even : ∀ x, g (-x) = g x)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 :=
sorry

end g_of_1_equals_3_l548_54866


namespace range_of_a_l548_54858

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l548_54858


namespace adam_simon_distance_100_l548_54877

noncomputable def time_to_be_100_apart (x : ℝ) : Prop :=
  let distance_adam := 10 * x
  let distance_simon_east := 10 * x * (Real.sqrt 2 / 2)
  let distance_simon_south := 10 * x * (Real.sqrt 2 / 2)
  let total_eastward_separation := abs (distance_adam - distance_simon_east)
  let resultant_distance := Real.sqrt (total_eastward_separation^2 + distance_simon_south^2)
  resultant_distance = 100

theorem adam_simon_distance_100 : ∃ (x : ℝ), time_to_be_100_apart x ∧ x = 2 * Real.sqrt 2 := 
by
  sorry

end adam_simon_distance_100_l548_54877


namespace divide_5440_K_l548_54803

theorem divide_5440_K (a b c d : ℕ) 
  (h1 : 5440 = a + b + c + d)
  (h2 : 2 * b = 3 * a)
  (h3 : 3 * c = 5 * b)
  (h4 : 5 * d = 6 * c) : 
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 :=
by 
  sorry

end divide_5440_K_l548_54803


namespace equidistant_points_quadrants_l548_54821

theorem equidistant_points_quadrants (x y : ℝ)
  (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : abs x = abs y) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_points_quadrants_l548_54821


namespace minimum_value_range_l548_54814

noncomputable def f (a x : ℝ) : ℝ := abs (3 * x - 1) + a * x + 2

theorem minimum_value_range (a : ℝ) :
  (-3 ≤ a ∧ a ≤ 3) ↔ ∃ m, ∀ x, f a x ≥ m := sorry

end minimum_value_range_l548_54814


namespace second_divisor_correct_l548_54863

noncomputable def smallest_num: Nat := 1012
def known_divisors := [12, 18, 21, 28]
def lcm_divisors: Nat := 252 -- This is the LCM of 12, 18, 21, and 28.
def result: Nat := 14

theorem second_divisor_correct :
  ∃ (d : Nat), d ≠ 12 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ d ≠ 252 ∧ (smallest_num - 4) % d = 0 ∧ d = result :=
by
  sorry

end second_divisor_correct_l548_54863


namespace tennis_balls_in_each_container_l548_54817

theorem tennis_balls_in_each_container (initial_balls : ℕ) (half_gone : ℕ) (remaining_balls : ℕ) (containers : ℕ) 
  (h1 : initial_balls = 100) 
  (h2 : half_gone = initial_balls / 2)
  (h3 : remaining_balls = initial_balls - half_gone)
  (h4 : containers = 5) :
  remaining_balls / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l548_54817


namespace compare_a_b_c_l548_54839

noncomputable def a := Real.sin (Real.pi / 5)
noncomputable def b := Real.logb (Real.sqrt 2) (Real.sqrt 3)
noncomputable def c := (1 / 4)^(2 / 3)

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l548_54839


namespace directrix_parabola_l548_54823

theorem directrix_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 8 * x^2 + 5) : 
  ∃ c : ℝ, ∀ x, y x = 8 * x^2 + 5 ∧ c = 159 / 32 :=
by
  use 159 / 32
  repeat { sorry }

end directrix_parabola_l548_54823


namespace books_at_end_l548_54869

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l548_54869


namespace at_least_one_not_less_than_four_l548_54856

theorem at_least_one_not_less_than_four 
( m n t : ℝ ) 
( h_m : 0 < m ) 
( h_n : 0 < n ) 
( h_t : 0 < t ) : 
∃ a, ( a = m + 4 / n ∨ a = n + 4 / t ∨ a = t + 4 / m ) ∧ 4 ≤ a :=
sorry

end at_least_one_not_less_than_four_l548_54856


namespace problem1_problem2_problem3_problem4_l548_54801

theorem problem1 : 0.175 / 0.25 / 4 = 0.175 := by
  sorry

theorem problem2 : 1.4 * 99 + 1.4 = 140 := by 
  sorry

theorem problem3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by
  sorry

theorem problem4 : (3.2 + 0.16) / 0.8 = 4.2 := by
  sorry

end problem1_problem2_problem3_problem4_l548_54801


namespace quadrilateral_diagonals_l548_54828

-- Define the points of the quadrilateral
variables {A B C D P Q R S : ℝ × ℝ}

-- Define the midpoints condition
def is_midpoint (M : ℝ × ℝ) (X Y : ℝ × ℝ) := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Define the lengths squared condition
def dist_sq (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Main theorem to prove
theorem quadrilateral_diagonals (hP : is_midpoint P A B) (hQ : is_midpoint Q B C)
  (hR : is_midpoint R C D) (hS : is_midpoint S D A) :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
by
  sorry

end quadrilateral_diagonals_l548_54828


namespace remainder_of_product_divided_by_7_l548_54831

theorem remainder_of_product_divided_by_7 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 2 :=
by
  sorry

end remainder_of_product_divided_by_7_l548_54831


namespace last_three_digits_product_l548_54822

theorem last_three_digits_product (a b c : ℕ) 
  (h1 : (a + b) % 10 = c % 10) 
  (h2 : (b + c) % 10 = a % 10) 
  (h3 : (c + a) % 10 = b % 10) :
  (a * b * c) % 1000 = 250 ∨ (a * b * c) % 1000 = 500 ∨ (a * b * c) % 1000 = 750 ∨ (a * b * c) % 1000 = 0 := 
by
  sorry

end last_three_digits_product_l548_54822


namespace bathroom_new_area_l548_54807

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l548_54807


namespace sequence_an_eq_n_l548_54816

theorem sequence_an_eq_n (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, n ≥ 1 → a n > 0) 
  (h₁ : ∀ n, n ≥ 1 → a n + 1 / 2 = Real.sqrt (2 * S n + 1 / 4)) : 
  ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end sequence_an_eq_n_l548_54816


namespace radish_patch_size_l548_54802

theorem radish_patch_size (R P : ℕ) (h1 : P = 2 * R) (h2 : P / 6 = 5) : R = 15 := by
  sorry

end radish_patch_size_l548_54802


namespace daily_profit_at_45_selling_price_for_1200_profit_l548_54849

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l548_54849


namespace queen_middle_school_teachers_l548_54845

theorem queen_middle_school_teachers
  (students : ℕ) 
  (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 6)
  (h_classes_per_teacher : classes_per_teacher = 5)
  (h_students_per_class : students_per_class = 25) : 
  (students * classes_per_student / students_per_class) / classes_per_teacher = 72 :=
by
  sorry

end queen_middle_school_teachers_l548_54845


namespace polar_circle_equation_l548_54847

theorem polar_circle_equation {r : ℝ} {phi : ℝ} {rho theta : ℝ} :
  (r = 2) → (phi = π / 3) → (rho = 4 * Real.cos (theta - π / 3)) :=
by
  intros hr hphi
  sorry

end polar_circle_equation_l548_54847


namespace ethanol_concentration_l548_54885

theorem ethanol_concentration
  (w1 : ℕ) (c1 : ℝ) (w2 : ℕ) (c2 : ℝ)
  (hw1 : w1 = 400) (hc1 : c1 = 0.30)
  (hw2 : w2 = 600) (hc2 : c2 = 0.80) :
  (c1 * w1 + c2 * w2) / (w1 + w2) = 0.60 := 
by
  sorry

end ethanol_concentration_l548_54885


namespace tom_killed_enemies_l548_54889

-- Define the number of points per enemy
def points_per_enemy : ℝ := 10

-- Define the bonus threshold and bonus factor
def bonus_threshold : ℝ := 100
def bonus_factor : ℝ := 1.5

-- Define the total score achieved by Tom
def total_score : ℝ := 2250

-- Define the number of enemies killed by Tom
variable (E : ℝ)

-- The proof goal
theorem tom_killed_enemies 
  (h1 : E ≥ bonus_threshold)
  (h2 : bonus_factor * points_per_enemy * E = total_score) : 
  E = 150 :=
sorry

end tom_killed_enemies_l548_54889


namespace min_candidates_for_same_score_l548_54811

theorem min_candidates_for_same_score :
  (∃ S : ℕ, S ≥ 25 ∧ (∀ elect : Fin S → Fin 12, ∃ s : Fin 12, ∃ a b c : Fin S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ elect a = s ∧ elect b = s ∧ elect c = s)) := 
sorry

end min_candidates_for_same_score_l548_54811


namespace total_carpet_area_correct_l548_54835

-- Define dimensions of the rooms
def room1_width : ℝ := 12
def room1_length : ℝ := 15
def room2_width : ℝ := 7
def room2_length : ℝ := 9
def room3_width : ℝ := 10
def room3_length : ℝ := 11

-- Define the areas of the rooms
def room1_area : ℝ := room1_width * room1_length
def room2_area : ℝ := room2_width * room2_length
def room3_area : ℝ := room3_width * room3_length

-- Total carpet area
def total_carpet_area : ℝ := room1_area + room2_area + room3_area

-- The theorem to prove
theorem total_carpet_area_correct :
  total_carpet_area = 353 :=
sorry

end total_carpet_area_correct_l548_54835


namespace second_integer_is_ninety_point_five_l548_54876

theorem second_integer_is_ninety_point_five
  (n : ℝ)
  (first_integer fourth_integer : ℝ)
  (h1 : first_integer = n - 2)
  (h2 : fourth_integer = n + 1)
  (h_sum : first_integer + fourth_integer = 180) :
  n = 90.5 :=
by
  -- sorry to skip the proof
  sorry

end second_integer_is_ninety_point_five_l548_54876


namespace sum_reciprocals_of_partial_fractions_l548_54879

noncomputable def f (s : ℝ) : ℝ := s^3 - 20 * s^2 + 125 * s - 500

theorem sum_reciprocals_of_partial_fractions :
  ∀ (p q r A B C : ℝ),
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    f p = 0 ∧ f q = 0 ∧ f r = 0 ∧
    (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
      (1 / f s = A / (s - p) + B / (s - q) + C / (s - r))) →
    1 / A + 1 / B + 1 / C = 720 :=
sorry

end sum_reciprocals_of_partial_fractions_l548_54879


namespace tory_sold_each_toy_gun_for_l548_54846

theorem tory_sold_each_toy_gun_for :
  ∃ (x : ℤ), 8 * 18 = 7 * x + 4 ∧ x = 20 := 
by
  use 20
  constructor
  · sorry
  · sorry

end tory_sold_each_toy_gun_for_l548_54846


namespace continued_fraction_l548_54861

theorem continued_fraction {w x y : ℕ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y)
  (h_eq : (97:ℚ) / 19 = w + 1 / (x + 1 / y)) : w + x + y = 16 :=
sorry

end continued_fraction_l548_54861


namespace find_number_l548_54812

theorem find_number (x : ℕ) (h : x + 5 * 8 = 340) : x = 300 :=
sorry

end find_number_l548_54812


namespace polygon_diagonals_l548_54870

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l548_54870


namespace lcm_ac_least_value_l548_54857

theorem lcm_ac_least_value (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : 
  Nat.lcm a c = 30 :=
sorry

end lcm_ac_least_value_l548_54857


namespace part1_max_value_l548_54810

variable (f : ℝ → ℝ)
def is_maximum (y : ℝ) := ∀ x : ℝ, f x ≤ y

theorem part1_max_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + m*x + 1) :
  m = 0 → (exists y, is_maximum f y ∧ y = 1) := 
sorry

end part1_max_value_l548_54810


namespace remainder_division_1614_254_eq_90_l548_54867

theorem remainder_division_1614_254_eq_90 :
  ∀ (x : ℕ) (R : ℕ),
    1614 - x = 1360 →
    x * 6 + R = 1614 →
    0 ≤ R →
    R < x →
    R = 90 := 
by
  intros x R h_diff h_div h_nonneg h_lt
  sorry

end remainder_division_1614_254_eq_90_l548_54867


namespace distinct_collections_proof_l548_54890

noncomputable def distinct_collections_count : ℕ := 240

theorem distinct_collections_proof : distinct_collections_count = 240 := by
  sorry

end distinct_collections_proof_l548_54890


namespace min_value_expression_l548_54886

theorem min_value_expression (x y z : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : z > 1) : ∃ C, C = 12 ∧
  ∀ (x y z : ℝ), x > 1 → y > 1 → z > 1 → (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ C := by
  sorry

end min_value_expression_l548_54886


namespace revenue_percentage_change_l548_54830

theorem revenue_percentage_change (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  let P_new := 1.30 * P
  let S_new := 0.80 * S
  let R := P * S
  let R_new := P_new * S_new
  (R_new - R) / R * 100 = 4 := by
  sorry

end revenue_percentage_change_l548_54830


namespace angle_B_eq_pi_div_3_l548_54898

variables {A B C : ℝ} {a b c : ℝ}

/-- Given an acute triangle ABC, where sides a, b, c are opposite the angles A, B, and C respectively, 
    and given the condition b cos C + sqrt 3 * b sin C = a + c, prove that B = π / 3. -/
theorem angle_B_eq_pi_div_3 
  (h : ∀ (A B C : ℝ), 0 < A ∧ A < π / 2  ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (cond : b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c) :
  B = π / 3 := 
sorry

end angle_B_eq_pi_div_3_l548_54898


namespace purchasing_methods_count_l548_54862

theorem purchasing_methods_count :
  ∃ n, n = 6 ∧
    ∃ (x y : ℕ), 
      60 * x + 70 * y ≤ 500 ∧
      x ≥ 3 ∧
      y ≥ 2 :=
sorry

end purchasing_methods_count_l548_54862


namespace red_car_speed_l548_54875

noncomputable def speed_blue : ℕ := 80
noncomputable def speed_green : ℕ := 8 * speed_blue
noncomputable def speed_red : ℕ := 2 * speed_green

theorem red_car_speed : speed_red = 1280 := by
  unfold speed_red
  unfold speed_green
  unfold speed_blue
  sorry

end red_car_speed_l548_54875


namespace sum_mod_five_l548_54887

theorem sum_mod_five {n : ℕ} (h_pos : 0 < n) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ ¬ (∃ k : ℕ, n = 4 * k) :=
sorry

end sum_mod_five_l548_54887


namespace triangle_area_example_l548_54873

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example : 
  triangle_area (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end triangle_area_example_l548_54873


namespace student_A_incorrect_l548_54881

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (cx, cy) := center
  let (px, py) := point
  (px - cx)^2 + (py - cy)^2 = radius^2

def center : ℝ × ℝ := (2, -3)
def radius : ℝ := 5
def point_A : ℝ × ℝ := (-2, -1)
def point_D : ℝ × ℝ := (5, 1)

theorem student_A_incorrect :
  ¬ is_on_circle center radius point_A ∧ is_on_circle center radius point_D :=
by
  sorry

end student_A_incorrect_l548_54881


namespace values_of_a_for_single_root_l548_54878

theorem values_of_a_for_single_root (a : ℝ) :
  (∃ (x : ℝ), ax^2 - 4 * x + 2 = 0) ∧ (∀ (x1 x2 : ℝ), ax^2 - 4 * x1 + 2 = 0 → ax^2 - 4 * x2 + 2 = 0 → x1 = x2) ↔ a = 0 ∨ a = 2 :=
sorry

end values_of_a_for_single_root_l548_54878


namespace trapezoid_perimeter_l548_54813

noncomputable def semiCircularTrapezoidPerimeter (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2) : ℝ :=
-((x^2) / 8) + 2 * x + 32

theorem trapezoid_perimeter 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2)
  (r : ℝ) 
  (h_r : r = 8) 
  (AB : ℝ) 
  (h_AB : AB = 2 * r)
  (CD_on_circumference : true) :
  semiCircularTrapezoidPerimeter x hx = -((x^2) / 8) + 2 * x + 32 :=   
sorry

end trapezoid_perimeter_l548_54813


namespace houses_distance_l548_54836

theorem houses_distance (num_houses : ℕ) (total_length : ℝ) (at_both_ends : Bool) 
  (h1: num_houses = 6) (h2: total_length = 11.5) (h3: at_both_ends = true) : 
  total_length / (num_houses - 1) = 2.3 := 
by
  sorry

end houses_distance_l548_54836


namespace graph_function_quadrant_l548_54818

theorem graph_function_quadrant (x y : ℝ): 
  (∀ x : ℝ, y = -x + 2 → (x < 0 → y ≠ -3 + - x)) := 
sorry

end graph_function_quadrant_l548_54818


namespace probability_of_spade_or_king_l548_54888

open Classical

-- Pack of cards containing 52 cards
def total_cards := 52

-- Number of spades in the deck
def num_spades := 13

-- Number of kings in the deck
def num_kings := 4

-- Number of overlap (king of spades)
def num_king_of_spades := 1

-- Total favorable outcomes
def total_favorable_outcomes := num_spades + num_kings - num_king_of_spades

-- Probability of drawing a spade or a king
def probability_spade_or_king := (total_favorable_outcomes : ℚ) / total_cards

theorem probability_of_spade_or_king : probability_spade_or_king = 4 / 13 := by
  sorry

end probability_of_spade_or_king_l548_54888


namespace jayson_age_l548_54804

/-- When Jayson is a certain age J, his dad is four times his age,
    and his mom is 2 years younger than his dad. Jayson's mom was
    28 years old when he was born. Prove that Jayson is 10 years old
    when his dad is four times his age. -/
theorem jayson_age {J : ℕ} (h1 : ∀ J, J > 0 → J * 4 < J + 4) 
                   (h2 : ∀ J, (4 * J - 2) = J + 28) 
                   (h3 : J - (4 * J - 28) = 0): 
                   J = 10 :=
by 
  sorry

end jayson_age_l548_54804


namespace tan_half_sum_pi_over_four_l548_54880

-- Define the problem conditions
variable (α : ℝ)
variable (h_cos : Real.cos α = -4 / 5)
variable (h_quad : α > π ∧ α < 3 * π / 2)

-- Define the theorem to prove
theorem tan_half_sum_pi_over_four (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_quad : α > π ∧ α < 3 * π / 2) :
  Real.tan (π / 4 + α / 2) = -1 / 2 := sorry

end tan_half_sum_pi_over_four_l548_54880


namespace triangle_in_base_7_l548_54855

theorem triangle_in_base_7 (triangle : ℕ) 
  (h1 : (triangle + 6) % 7 = 0) : 
  triangle = 1 := 
sorry

end triangle_in_base_7_l548_54855


namespace money_left_after_shopping_l548_54838

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end money_left_after_shopping_l548_54838


namespace custom_op_eval_l548_54892

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 5 * a + 2 * b - 1

-- State the required proof problem
theorem custom_op_eval : custom_op (-4) 6 = -9 := 
by
  -- use sorry to skip the proof
  sorry

end custom_op_eval_l548_54892


namespace find_ax_plus_a_negx_l548_54864

theorem find_ax_plus_a_negx
  (a : ℝ) (x : ℝ)
  (h₁ : a > 0)
  (h₂ : a^(x/2) + a^(-x/2) = 5) :
  a^x + a^(-x) = 23 :=
by
  sorry

end find_ax_plus_a_negx_l548_54864


namespace bill_amount_each_person_shared_l548_54851

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def num_people : ℝ := 7.00

noncomputable def tip : ℝ := tip_percentage * total_bill
noncomputable def total_bill_with_tip : ℝ := total_bill + tip
noncomputable def amount_each_person_pays : ℝ := total_bill_with_tip / num_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 21.84 := by
  -- proof goes here
  sorry

end bill_amount_each_person_shared_l548_54851


namespace complement_correct_l548_54809

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the set of real numbers such that -1 ≤ x < 2
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

-- The proof statement: the complement of A in U is the expected set
theorem complement_correct : (U \ A) = complement_U_A := 
by
  sorry

end complement_correct_l548_54809


namespace arithmetic_series_first_term_l548_54832

theorem arithmetic_series_first_term (a d : ℚ) 
  (h1 : 15 * (2 * a + 29 * d) = 450) 
  (h2 : 15 * (2 * a + 89 * d) = 1950) : 
  a = -55 / 6 :=
by 
  sorry

end arithmetic_series_first_term_l548_54832


namespace percentage_of_bags_not_sold_l548_54860

theorem percentage_of_bags_not_sold
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : sold_monday = 25)
  (h_tuesday : sold_tuesday = 70)
  (h_wednesday : sold_wednesday = 100)
  (h_thursday : sold_thursday = 110)
  (h_friday : sold_friday = 145) : 
  (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) * 100 / initial_stock = 25 :=
by
  sorry

end percentage_of_bags_not_sold_l548_54860


namespace new_class_mean_l548_54893

theorem new_class_mean 
  (n1 n2 : ℕ) (mean1 mean2 : ℚ) 
  (h1 : n1 = 24) (h2 : n2 = 8) 
  (h3 : mean1 = 85/100) (h4 : mean2 = 90/100) :
  (n1 * mean1 + n2 * mean2) / (n1 + n2) = 345/400 :=
by
  rw [h1, h2, h3, h4]
  sorry

end new_class_mean_l548_54893


namespace arithmetic_sequence_product_l548_54829

theorem arithmetic_sequence_product (a d : ℕ) :
  (a + 7 * d = 20) → (d = 2) → ((a + d) * (a + 2 * d) = 80) :=
by
  intros h₁ h₂
  sorry

end arithmetic_sequence_product_l548_54829


namespace financial_loss_example_l548_54805

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := 
  P * (1 + r * t)

theorem financial_loss_example :
  let P := 10000
  let r1 := 0.06
  let r2 := 0.05
  let t := 3
  let n := 4
  let A1 := compound_interest P r1 n t
  let A2 := simple_interest P r2 t
  abs (A1 - A2 - 456.18) < 0.01 := by
    sorry

end financial_loss_example_l548_54805


namespace find_c_l548_54859

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c_l548_54859


namespace five_peso_coins_count_l548_54840

theorem five_peso_coins_count (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 5 * y = 440) (h3 : x = 24 ∨ y = 24) : y = 24 :=
by sorry

end five_peso_coins_count_l548_54840


namespace ellipse_foci_y_axis_iff_l548_54837

theorem ellipse_foci_y_axis_iff (m n : ℝ) (h : m > n ∧ n > 0) :
  (m > n ∧ n > 0) ↔ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → ∃ a b : ℝ, a^2 - b^2 = 1 ∧ x^2/b^2 + y^2/a^2 = 1 ∧ a > b) :=
sorry

end ellipse_foci_y_axis_iff_l548_54837


namespace find_a_and_b_l548_54824

theorem find_a_and_b (a b : ℝ) (h1 : b ≠ 0) 
  (h2 : (ab = a + b ∨ ab = a - b ∨ ab = a / b) 
  ∧ (a + b = a - b ∨ a + b = a / b) 
  ∧ (a - b = a / b)) : 
  (a = 1 / 2 ∨ a = -1 / 2) ∧ b = -1 := by
  sorry

end find_a_and_b_l548_54824


namespace simplify_T_l548_54833

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6 * (x + 2)^5 + 15 * (x + 2)^4 + 20 * (x + 2)^3 + 15 * (x + 2)^2 + 6 * (x + 2) + 1 = (x + 3)^6 :=
by
  sorry

end simplify_T_l548_54833


namespace abs_of_negative_l548_54865

theorem abs_of_negative (a : ℝ) (h : a < 0) : |a| = -a :=
sorry

end abs_of_negative_l548_54865


namespace max_two_terms_eq_one_l548_54808

theorem max_two_terms_eq_one (a b c x y z : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z) :
  ∀ (P : ℕ → ℝ), -- Define P(i) as given expressions
  ((P 1 = a * x + b * y + c * z) ∧
   (P 2 = a * x + b * z + c * y) ∧
   (P 3 = a * y + b * x + c * z) ∧
   (P 4 = a * y + b * z + c * x) ∧
   (P 5 = a * z + b * x + c * y) ∧
   (P 6 = a * z + b * y + c * x)) →
  (P 1 = 1 ∨ P 2 = 1 ∨ P 3 = 1 ∨ P 4 = 1 ∨ P 5 = 1 ∨ P 6 = 1) →
  (∃ i j, i ≠ j ∧ P i = 1 ∧ P j = 1) →
  ¬(∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ P i = 1 ∧ P j = 1 ∧ P k = 1) :=
sorry

end max_two_terms_eq_one_l548_54808


namespace largest_divisor_of_five_consecutive_odds_l548_54874

theorem largest_divisor_of_five_consecutive_odds (n : ℕ) (hn : n % 2 = 0) :
    ∃ d, d = 15 ∧ ∀ m, (m = (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) → d ∣ m :=
sorry

end largest_divisor_of_five_consecutive_odds_l548_54874


namespace C_must_be_2_l548_54844

-- Define the given digits and their sum conditions
variables (A B C D : ℤ)

-- The sum of known digits for the first number
def sum1_known_digits := 7 + 4 + 5 + 2

-- The sum of known digits for the second number
def sum2_known_digits := 3 + 2 + 6 + 5

-- The first number must be divisible by 3
def divisible_by_3 (n : ℤ) : Prop := n % 3 = 0

-- Conditions for the divisibility by 3 of both numbers
def conditions := divisible_by_3 (sum1_known_digits + A + B + D) ∧ 
                  divisible_by_3 (sum2_known_digits + A + B + C)

-- The statement of the theorem
theorem C_must_be_2 (A B D : ℤ) (h : conditions A B 2 D) : C = 2 :=
  sorry

end C_must_be_2_l548_54844


namespace fraction_over_65_l548_54894

def num_people_under_21 := 33
def fraction_under_21 := 3 / 7
def total_people (N : ℕ) := N > 50 ∧ N < 100
def num_people (N : ℕ) := num_people_under_21 = fraction_under_21 * N

theorem fraction_over_65 (N : ℕ) : 
  total_people N → num_people N → N = 77 ∧ ∃ x, (x / 77) = x / 77 :=
by
  intro hN hnum
  sorry

end fraction_over_65_l548_54894


namespace fg_of_2_l548_54899

def f (x : ℤ) : ℤ := 4 * x + 3
def g (x : ℤ) : ℤ := x ^ 3 + 1

theorem fg_of_2 : f (g 2) = 39 := by
  sorry

end fg_of_2_l548_54899


namespace integer_roots_sum_abs_eq_94_l548_54842

theorem integer_roots_sum_abs_eq_94 {a b c m : ℤ} :
  (∃ m, (x : ℤ) * (x : ℤ) * (x : ℤ) - 2013 * (x : ℤ) + m = 0 ∧ a + b + c = 0 ∧ ab + bc + ac = -2013) →
  |a| + |b| + |c| = 94 :=
sorry

end integer_roots_sum_abs_eq_94_l548_54842


namespace hyperbola_problem_l548_54895

-- Given the conditions of the hyperbola
def hyperbola (x y: ℝ) (b: ℝ) : Prop := (x^2) / 4 - (y^2) / (b^2) = 1 ∧ b > 0

-- Asymptote condition
def asymptote (b: ℝ) : Prop := (b / 2) = (Real.sqrt 6 / 2)

-- Foci, point P condition
def foci_and_point (PF1 PF2: ℝ) : Prop := PF1 / PF2 = 3 / 1 ∧ PF1 - PF2 = 4

-- Math proof problem
theorem hyperbola_problem (b PF1 PF2: ℝ) (P: ℝ × ℝ) :
  hyperbola P.1 P.2 b ∧ asymptote b ∧ foci_and_point PF1 PF2 →
  |PF1 + PF2| = 2 * Real.sqrt 10 :=
by
  sorry

end hyperbola_problem_l548_54895


namespace find_original_number_l548_54882

theorem find_original_number (h1 : 268 * 74 = 19732) (h2 : 2.68 * x = 1.9832) : x = 0.74 :=
sorry

end find_original_number_l548_54882


namespace geometric_seq_comparison_l548_54897

def geometric_seq_positive (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n+1) = a n * q

theorem geometric_seq_comparison (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq_positive a q) (h2 : q ≠ 1) (h3 : ∀ n, a n > 0) (h4 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_seq_comparison_l548_54897


namespace gcd_pair_sum_ge_prime_l548_54871

theorem gcd_pair_sum_ge_prime
  (n : ℕ)
  (h_prime: Prime (2*n - 1))
  (a : Fin n → ℕ)
  (h_distinct: ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / Nat.gcd (a i) (a j) ≥ 2*n - 1 := sorry

end gcd_pair_sum_ge_prime_l548_54871


namespace average_marbles_of_other_colors_l548_54896

theorem average_marbles_of_other_colors
  (clear_percentage : ℝ) (black_percentage : ℝ) (total_marbles_taken : ℕ)
  (h1 : clear_percentage = 0.4) (h2 : black_percentage = 0.2) :
  (total_marbles_taken : ℝ) * (1 - clear_percentage - black_percentage) = 2 :=
by
  sorry

end average_marbles_of_other_colors_l548_54896


namespace negation_universal_proposition_l548_54825

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := 
by 
  sorry

end negation_universal_proposition_l548_54825


namespace efficiency_ratio_l548_54820

-- Define the work efficiencies
def EA : ℚ := 1 / 12
def EB : ℚ := 1 / 24
def EAB : ℚ := 1 / 8

-- State the theorem
theorem efficiency_ratio (EAB_eq : EAB = EA + EB) : (EA / EB) = 2 := by
  -- Insert proof here
  sorry

end efficiency_ratio_l548_54820


namespace probability_of_odd_sum_given_even_product_l548_54854

-- Define a function to represent the probability of an event given the conditions
noncomputable def conditional_probability_odd_sum_even_product (dice : Fin 5 → Fin 8) : ℚ :=
  if h : (∃ i, (dice i).val % 2 = 0)  -- At least one die is even (product is even)
  then (1/2) / (31/32)  -- Probability of odd sum given even product
  else 0  -- If product is not even (not possible under conditions)

theorem probability_of_odd_sum_given_even_product :
  ∀ (dice : Fin 5 → Fin 8),
  conditional_probability_odd_sum_even_product dice = 16/31 :=
sorry  -- Proof omitted

end probability_of_odd_sum_given_even_product_l548_54854


namespace problem1_problem2_l548_54848

theorem problem1 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) :
  (2 < x1 + x2 ∧ x1 + x2 < 6) ∧ |x1 - x2| < 2 :=
by
  sorry

theorem problem2 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 - x + 1) :
  |x1 - x2| < |f x1 - f x2| ∧ |f x1 - f x2| < 5 * |x1 - x2| :=
by
  sorry

end problem1_problem2_l548_54848


namespace student_weight_l548_54852

-- Define the weights of the student and sister
variables (S R : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := S - 5 = 1.25 * R
def condition2 : Prop := S + R = 104

-- The theorem we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 60 := 
by
  sorry

end student_weight_l548_54852


namespace mandy_book_length_l548_54827

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end mandy_book_length_l548_54827


namespace certain_number_l548_54853

theorem certain_number (a x : ℝ) (h1 : a / x * 2 = 12) (h2 : x = 0.1) : a = 0.6 := 
by
  sorry

end certain_number_l548_54853


namespace evaluate_expression_at_x_l548_54819

theorem evaluate_expression_at_x (x : ℝ) (h : x = Real.sqrt 2 - 3) : 
  (3 * x / (x^2 - 9)) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end evaluate_expression_at_x_l548_54819


namespace calculate_expression_l548_54850

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 2 * x + 4

theorem calculate_expression : f (g 2) - g (f 2) = 49 := by
  sorry

end calculate_expression_l548_54850


namespace tangent_line_eq_l548_54815

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x^2 + 1) : 
  (x = -1 ∧ y = 3) → (4 * x + y + 1 = 0) :=
by
  intros
  sorry

end tangent_line_eq_l548_54815


namespace inequality_solution_set_range_of_m_l548_54868

-- Proof Problem 1
theorem inequality_solution_set :
  {x : ℝ | -2 < x ∧ x < 4} = { x : ℝ | 2 * x^2 - 4 * x - 16 < 0 } :=
sorry

-- Proof Problem 2
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 :=
  sorry

end inequality_solution_set_range_of_m_l548_54868


namespace garrison_reinforcement_l548_54834

/-- A garrison has initial provisions for 2000 men for 65 days. 
    After 15 days, reinforcement arrives and the remaining provisions last for 20 more days. 
    The size of the reinforcement is 3000 men.  -/
theorem garrison_reinforcement (P : ℕ) (M1 M2 D1 D2 D3 R : ℕ) 
  (h1 : M1 = 2000) (h2 : D1 = 65) (h3 : D2 = 15) (h4 : D3 = 20) 
  (h5 : P = M1 * D1) (h6 : P - M1 * D2 = (M1 + R) * D3) : 
  R = 3000 := 
sorry

end garrison_reinforcement_l548_54834


namespace simplify_fraction_l548_54826

theorem simplify_fraction :
  (6 * x ^ 3 + 13 * x ^ 2 + 15 * x - 25) / (2 * x ^ 3 + 4 * x ^ 2 + 4 * x - 10) =
  (6 * x - 5) / (2 * x - 2) :=
by
  sorry

end simplify_fraction_l548_54826


namespace smallest_base_to_represent_124_with_three_digits_l548_54843

theorem smallest_base_to_represent_124_with_three_digits : 
  ∃ (b : ℕ), b^2 ≤ 124 ∧ 124 < b^3 ∧ ∀ c, (c^2 ≤ 124 ∧ 124 < c^3) → (5 ≤ c) :=
by
  sorry

end smallest_base_to_represent_124_with_three_digits_l548_54843


namespace stock_status_after_limit_moves_l548_54872

theorem stock_status_after_limit_moves (initial_value : ℝ) (h₁ : initial_value = 1)
  (limit_up_factor : ℝ) (h₂ : limit_up_factor = 1 + 0.10)
  (limit_down_factor : ℝ) (h₃ : limit_down_factor = 1 - 0.10) :
  (limit_up_factor^5 * limit_down_factor^5) < initial_value :=
by
  sorry

end stock_status_after_limit_moves_l548_54872


namespace exists_geometric_arithmetic_progressions_l548_54891

theorem exists_geometric_arithmetic_progressions (n : ℕ) (hn : n > 3) :
  ∃ (x y : ℕ → ℕ),
  (∀ m < n, x (m + 1) = (1 + ε)^m ∧ y (m + 1) = (1 + (m + 1) * ε - δ)) ∧
  ∀ m < n, x m < y m ∧ y m < x (m + 1) :=
by
  sorry

end exists_geometric_arithmetic_progressions_l548_54891


namespace birds_find_more_than_half_millet_on_thursday_l548_54800

def millet_on_day (n : ℕ) : ℝ :=
  2 - 2 * (0.7 ^ n)

def more_than_half_millet (day : ℕ) : Prop :=
  millet_on_day day > 1

theorem birds_find_more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by
  sorry

end birds_find_more_than_half_millet_on_thursday_l548_54800


namespace g_inv_g_inv_14_l548_54806

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l548_54806


namespace perfect_square_trinomial_l548_54883

theorem perfect_square_trinomial (m : ℝ) : (∃ b : ℝ, (x^2 - 6 * x + m) = (x + b) ^ 2) → m = 9 :=
by
  sorry

end perfect_square_trinomial_l548_54883
