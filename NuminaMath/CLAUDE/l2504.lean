import Mathlib

namespace triple_base_double_exponent_l2504_250443

theorem triple_base_double_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end triple_base_double_exponent_l2504_250443


namespace chimps_in_old_cage_l2504_250421

/-- The number of chimps staying in the old cage is equal to the total number of chimps minus the number of chimps being moved. -/
theorem chimps_in_old_cage (total_chimps moving_chimps : ℕ) :
  total_chimps ≥ moving_chimps →
  total_chimps - moving_chimps = total_chimps - moving_chimps :=
by
  sorry

#check chimps_in_old_cage 45 18

end chimps_in_old_cage_l2504_250421


namespace girl_travel_distance_l2504_250495

/-- 
Given a constant speed and time, calculates the distance traveled.
-/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- 
Theorem: A girl traveling at 4 m/s for 32 seconds covers a distance of 128 meters.
-/
theorem girl_travel_distance : 
  distance_traveled 4 32 = 128 := by
  sorry

end girl_travel_distance_l2504_250495


namespace area_of_K_l2504_250411

/-- The set K in the plane Cartesian coordinate system xOy -/
def K : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

/-- The area of set K -/
theorem area_of_K : MeasureTheory.volume K = 24 := by
  sorry

end area_of_K_l2504_250411


namespace david_boxes_l2504_250475

/-- Given a total number of dogs and the number of dogs per box, 
    calculate the number of boxes needed. -/
def calculate_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) : ℕ :=
  total_dogs / dogs_per_box

/-- Theorem stating that given 28 total dogs and 4 dogs per box, 
    the number of boxes is 7. -/
theorem david_boxes : calculate_boxes 28 4 = 7 := by
  sorry

end david_boxes_l2504_250475


namespace distribute_six_balls_three_boxes_l2504_250476

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  n_boxes ^ n_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 3^6 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 3^6 := by
  sorry

end distribute_six_balls_three_boxes_l2504_250476


namespace smallest_non_prime_non_square_with_large_factors_l2504_250460

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∀ n : ℕ, n > 0 → 
  (¬ is_prime n) → 
  (¬ is_square n) → 
  (smallest_prime_factor n ≥ 60) → 
  n ≥ 4087 := by
  sorry

end smallest_non_prime_non_square_with_large_factors_l2504_250460


namespace rectangle_area_l2504_250459

/-- Given a rectangle with diagonal length y and length three times its width, 
    prove that its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (h : y > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

#check rectangle_area

end rectangle_area_l2504_250459


namespace total_weight_of_goods_l2504_250451

theorem total_weight_of_goods (x : ℝ) 
  (h1 : (x - 10) / 7 = (x + 5) / 8) : x = 115 := by
  sorry

#check total_weight_of_goods

end total_weight_of_goods_l2504_250451


namespace moses_extra_amount_l2504_250498

def total_amount : ℝ := 50
def moses_percentage : ℝ := 0.4

theorem moses_extra_amount :
  let moses_share := moses_percentage * total_amount
  let remainder := total_amount - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by sorry

end moses_extra_amount_l2504_250498


namespace milk_replacement_theorem_l2504_250441

/-- The fraction of original substance remaining after one replacement operation -/
def replacement_fraction : ℝ := 0.8

/-- The number of replacement operations -/
def num_operations : ℕ := 3

/-- The percentage of original substance remaining after multiple replacement operations -/
def remaining_percentage (f : ℝ) (n : ℕ) : ℝ := 100 * f^n

theorem milk_replacement_theorem :
  remaining_percentage replacement_fraction num_operations = 51.2 := by
  sorry

end milk_replacement_theorem_l2504_250441


namespace fold_line_equation_l2504_250400

/-- The perpendicular bisector of the line segment joining two points (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem fold_line_equation :
  perpendicular_bisector 5 3 1 (-1) = {p : ℝ × ℝ | p.2 = -p.1 + 4} := by
  sorry

end fold_line_equation_l2504_250400


namespace janet_total_earnings_l2504_250402

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℝ) (sculpture_rate : ℝ) (hours_worked : ℝ) 
                   (sculpture1_weight : ℝ) (sculpture2_weight : ℝ) : ℝ :=
  exterminator_rate * hours_worked + 
  sculpture_rate * (sculpture1_weight + sculpture2_weight)

/-- Proves that Janet's total earnings are $1640 given the specified conditions -/
theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end janet_total_earnings_l2504_250402


namespace no_common_solution_exists_l2504_250457

/-- A_{n}^k denotes the number of k-permutations of n elements -/
def A (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- C_{n}^k denotes the number of k-combinations of n elements -/
def C (n k : ℕ) : ℕ := Nat.choose n k

theorem no_common_solution_exists : ¬ ∃ (n : ℕ), n ≥ 3 ∧ 
  A (2*n) 3 = 2 * A (n+1) 4 ∧ 
  C (n+2) (n-2) + C (n+2) (n-3) = (A (n+3) 3) / 10 := by
  sorry

end no_common_solution_exists_l2504_250457


namespace innovation_cup_award_eligibility_l2504_250494

/-- Represents the "Innovation Cup" basketball competition rules and Xiao Ming's team's goal --/
theorem innovation_cup_award_eligibility 
  (total_games : ℕ) 
  (min_points_for_award : ℕ) 
  (points_per_win : ℕ) 
  (points_per_loss : ℕ) 
  (h1 : total_games = 8)
  (h2 : min_points_for_award = 12)
  (h3 : points_per_win = 2)
  (h4 : points_per_loss = 1)
  : ∀ x : ℕ, x ≤ total_games → 
    (x * points_per_win + (total_games - x) * points_per_loss ≥ min_points_for_award ↔ 
     2 * x + (8 - x) ≥ 12) :=
by sorry

end innovation_cup_award_eligibility_l2504_250494


namespace arithmetic_mean_problem_l2504_250433

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 22 + 5 + y) / 5 = 12 → y = 10 := by
sorry

end arithmetic_mean_problem_l2504_250433


namespace simplify_expression_l2504_250409

theorem simplify_expression (b c : ℝ) : 
  (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) * (7 * c^2) = 5040 * b^10 * c^2 := by
  sorry

end simplify_expression_l2504_250409


namespace secret_spread_reaches_target_l2504_250456

/-- Represents the number of students who know the secret on a given day -/
def secret_spread : ℕ → ℕ
| 0 => 4  -- Monday (day 0): Jessica + 3 friends
| 1 => 10 -- Tuesday (day 1): Previous + 2 * 3 new
| 2 => 22 -- Wednesday (day 2): Previous + 3 * 3 + 3 new
| n + 3 => secret_spread (n + 2) + 3 * (secret_spread (n + 2) - secret_spread (n + 1))

/-- The day when the secret reaches at least 7280 students -/
def target_day : ℕ := 9

theorem secret_spread_reaches_target :
  secret_spread target_day ≥ 7280 := by
  sorry


end secret_spread_reaches_target_l2504_250456


namespace delivery_driver_stops_l2504_250479

theorem delivery_driver_stops (initial_stops total_stops : ℕ) 
  (h1 : initial_stops = 3)
  (h2 : total_stops = 7) :
  total_stops - initial_stops = 4 := by
  sorry

end delivery_driver_stops_l2504_250479


namespace four_digit_sum_27_eq_3276_l2504_250468

/-- The number of four-digit whole numbers whose digits sum to 27 -/
def four_digit_sum_27 : ℕ :=
  (Finset.range 10).sum (fun a =>
    (Finset.range 10).sum (fun b =>
      (Finset.range 10).sum (fun c =>
        (Finset.range 10).sum (fun d =>
          if a ≥ 1 ∧ a + b + c + d = 27 then 1 else 0))))

theorem four_digit_sum_27_eq_3276 : four_digit_sum_27 = 3276 := by
  sorry

end four_digit_sum_27_eq_3276_l2504_250468


namespace cubic_monomial_properties_l2504_250413

/-- A cubic monomial with coefficient -2 using only variables x and y -/
def cubic_monomial (x y : ℝ) : ℝ := -2 * x^2 * y

theorem cubic_monomial_properties (x y : ℝ) :
  ∃ (a b c : ℕ), a + b + c = 3 ∧ cubic_monomial x y = -2 * x^a * y^b := by
  sorry

end cubic_monomial_properties_l2504_250413


namespace completing_square_result_l2504_250408

theorem completing_square_result (x : ℝ) : 
  x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
  sorry

end completing_square_result_l2504_250408


namespace inequality_of_cubes_l2504_250447

theorem inequality_of_cubes (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by
  sorry

end inequality_of_cubes_l2504_250447


namespace purchases_total_price_l2504_250410

/-- The total price of a refrigerator and a washing machine -/
def total_price (refrigerator_price washing_machine_price : ℕ) : ℕ :=
  refrigerator_price + washing_machine_price

/-- Theorem: The total price of the purchases is $7060 -/
theorem purchases_total_price :
  let refrigerator_price : ℕ := 4275
  let washing_machine_price : ℕ := refrigerator_price - 1490
  total_price refrigerator_price washing_machine_price = 7060 := by
sorry

end purchases_total_price_l2504_250410


namespace divisibility_by_24_l2504_250401

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end divisibility_by_24_l2504_250401


namespace cyclic_ratio_sum_geq_two_l2504_250483

theorem cyclic_ratio_sum_geq_two (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end cyclic_ratio_sum_geq_two_l2504_250483


namespace arithmetic_sequence_20th_term_l2504_250430

theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                            -- first term is 3
    a 1 = 7 →                            -- second term is 7
    a 19 = 79 :=                         -- 20th term (index 19) is 79
by
  sorry

end arithmetic_sequence_20th_term_l2504_250430


namespace cans_per_bag_l2504_250487

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 63) (h2 : total_bags = 7) :
  total_cans / total_bags = 9 := by
  sorry

end cans_per_bag_l2504_250487


namespace sqrt_50_between_consecutive_integers_product_l2504_250470

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ 
  Real.sqrt 50 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_product_l2504_250470


namespace arctan_sum_three_four_l2504_250486

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end arctan_sum_three_four_l2504_250486


namespace large_cube_surface_area_l2504_250499

-- Define the volume of a small cube
def small_cube_volume : ℝ := 512

-- Define the number of small cubes
def num_small_cubes : ℕ := 8

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem large_cube_surface_area :
  let small_side := side_length small_cube_volume
  let large_side := small_side * (num_small_cubes ^ (1/3))
  surface_area large_side = 1536 := by
  sorry

end large_cube_surface_area_l2504_250499


namespace p_bounds_l2504_250428

/-- Represents the minimum number of reconstructions needed to transform
    one triangulation into another for a convex n-gon. -/
def p (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds on p(n) for convex n-gons. -/
theorem p_bounds (n : ℕ) : 
  n ≥ 3 → 
  p n ≥ n - 3 ∧ 
  p n ≤ 2*n - 7 ∧ 
  (n ≥ 13 → p n ≤ 2*n - 10) := by sorry


end p_bounds_l2504_250428


namespace max_students_distribution_l2504_250465

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1008) (h2 : pencils = 928) :
  (Nat.gcd pens pencils : ℕ) = 16 := by sorry

end max_students_distribution_l2504_250465


namespace total_pencils_l2504_250481

theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : desk_added = 16) :
  drawer + desk_initial + desk_added = 78 :=
by sorry

end total_pencils_l2504_250481


namespace complement_of_M_in_U_l2504_250493

-- Define the universal set U
def U : Set ℤ := {-1, -2, -3, 0, 1}

-- Define set M
def M (a : ℤ) : Set ℤ := {-1, 0, a^2 + 1}

-- Theorem statement
theorem complement_of_M_in_U (a : ℤ) (h : M a ⊆ U) :
  U \ M a = {-2, -3} := by sorry

end complement_of_M_in_U_l2504_250493


namespace point_three_units_from_negative_two_l2504_250437

def point_on_number_line (x : ℝ) := True

theorem point_three_units_from_negative_two (A : ℝ) :
  point_on_number_line A →
  |A - (-2)| = 3 →
  A = -5 ∨ A = 1 := by
  sorry

end point_three_units_from_negative_two_l2504_250437


namespace intersection_of_M_and_N_l2504_250463

def M : Set ℝ := {2, 4, 6, 8}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l2504_250463


namespace largest_integer_divisibility_l2504_250488

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (m : ℕ), m > n → ¬(∃ (k : ℤ), (m - 2)^2 * (m + 1) = k * (2*m - 1))) ∧
  (∃ (k : ℤ), (n - 2)^2 * (n + 1) = k * (2*n - 1)) := by
  sorry

end largest_integer_divisibility_l2504_250488


namespace problem_solution_l2504_250454

theorem problem_solution : 
  (∃ x : ℚ, x - 2/11 = -1/3 ∧ x = -5/33) ∧ 
  (-2 - (-1/3 + 1/2) = -13/6) := by
sorry

end problem_solution_l2504_250454


namespace parallelogram_x_value_l2504_250425

/-- A parallelogram ABCD with specific properties -/
structure Parallelogram where
  x : ℝ
  area : ℝ
  h : x > 0
  angle : ℝ
  h_angle : angle = 30 * π / 180
  h_area : area = 35

/-- The theorem stating that x = 14 for the given parallelogram -/
theorem parallelogram_x_value (p : Parallelogram) : p.x = 14 := by
  sorry

end parallelogram_x_value_l2504_250425


namespace angle_is_120_degrees_l2504_250478

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_120_degrees (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 4)
  (h2 : b = (-1, 0))
  (h3 : (a.1 + 2 * b.1) * b.1 + (a.2 + 2 * b.2) * b.2 = 0) :
  angle_between_vectors a b = 2 * π / 3 := by sorry

end angle_is_120_degrees_l2504_250478


namespace max_value_theorem_l2504_250462

theorem max_value_theorem (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 200) :
  ∃ (max_value : ℝ), max_value = 30000 ∧ 
  ∀ (x y z w : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → 
  x + y + z + w = 200 → 2*x*y + 3*y*z + 4*z*w ≤ max_value :=
by sorry

end max_value_theorem_l2504_250462


namespace choose_three_cooks_from_ten_l2504_250473

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end choose_three_cooks_from_ten_l2504_250473


namespace four_points_probability_l2504_250497

-- Define a circle
def Circle : Type := Unit

-- Define a point on a circle
def Point (c : Circle) : Type := Unit

-- Define a function to choose n points uniformly at random on a circle
def chooseRandomPoints (c : Circle) (n : ℕ) : Type := 
  Fin n → Point c

-- Define a predicate for two points and the center forming an obtuse triangle
def isObtuse (c : Circle) (p1 p2 : Point c) : Prop := sorry

-- Define a function to calculate the probability of an event
def probability (event : Prop) : ℝ := sorry

-- The main theorem
theorem four_points_probability (c : Circle) :
  probability (∀ (points : chooseRandomPoints c 4),
    ∀ (i j : Fin 4), i ≠ j → ¬isObtuse c (points i) (points j)) = 1 / 64 := by
  sorry

end four_points_probability_l2504_250497


namespace pizza_slices_per_pizza_l2504_250458

theorem pizza_slices_per_pizza (num_people : ℕ) (slices_per_person : ℕ) (num_pizzas : ℕ) : 
  num_people = 18 → slices_per_person = 3 → num_pizzas = 6 →
  (num_people * slices_per_person) / num_pizzas = 9 := by
  sorry

end pizza_slices_per_pizza_l2504_250458


namespace chocolate_cost_in_dollars_l2504_250461

/-- The cost of the chocolate in cents -/
def chocolate_cost (money_in_pocket : ℕ) (borrowed : ℕ) (needed : ℕ) : ℕ :=
  money_in_pocket * 100 + borrowed + needed

theorem chocolate_cost_in_dollars :
  let money_in_pocket : ℕ := 4
  let borrowed : ℕ := 59
  let needed : ℕ := 41
  (chocolate_cost money_in_pocket borrowed needed) / 100 = 5 := by
  sorry

end chocolate_cost_in_dollars_l2504_250461


namespace jars_left_unpacked_eighty_jars_left_l2504_250445

/-- The number of jars left unpacked given the packing configuration and total jars --/
theorem jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
  (jars_per_box2 : ℕ) (num_boxes2 : ℕ) (total_jars : ℕ) : ℕ :=
  by
  have h1 : jars_per_box1 = 12 := by sorry
  have h2 : num_boxes1 = 10 := by sorry
  have h3 : jars_per_box2 = 10 := by sorry
  have h4 : num_boxes2 = 30 := by sorry
  have h5 : total_jars = 500 := by sorry
  
  let packed_jars := jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2
  
  have packed_eq : packed_jars = 420 := by sorry
  
  exact total_jars - packed_jars

/-- The main theorem stating that 80 jars will be left unpacked --/
theorem eighty_jars_left : jars_left_unpacked 12 10 10 30 500 = 80 := by sorry

end jars_left_unpacked_eighty_jars_left_l2504_250445


namespace fourth_power_sum_of_roots_l2504_250471

theorem fourth_power_sum_of_roots (r₁ r₂ r₃ r₄ : ℝ) : 
  (r₁^4 - r₁ - 504 = 0) → 
  (r₂^4 - r₂ - 504 = 0) → 
  (r₃^4 - r₃ - 504 = 0) → 
  (r₄^4 - r₄ - 504 = 0) → 
  r₁^4 + r₂^4 + r₃^4 + r₄^4 = 2016 := by
  sorry

end fourth_power_sum_of_roots_l2504_250471


namespace largest_five_digit_congruent_to_17_mod_28_l2504_250406

theorem largest_five_digit_congruent_to_17_mod_28 : ∃ x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  x ≡ 17 [MOD 28] ∧
  (∀ y : ℕ, (y ≥ 10000 ∧ y < 100000) ∧ y ≡ 17 [MOD 28] → y ≤ x) ∧
  x = 99947 := by
sorry

end largest_five_digit_congruent_to_17_mod_28_l2504_250406


namespace cyclist_speed_ratio_l2504_250419

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
  v₁ > v₂ →
  v₁ + v₂ = 20 →
  v₁ - v₂ = 5 →
  v₁ / v₂ = 5 / 3 :=
by
  sorry

end cyclist_speed_ratio_l2504_250419


namespace arithmetic_sequence_common_difference_l2504_250482

/-- The common difference of an arithmetic sequence with general term a_n = 5 - 4n is -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), (∀ n, a n = 5 - 4 * n) →
  ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = -4 :=
by sorry

end arithmetic_sequence_common_difference_l2504_250482


namespace simplest_quadratic_radical_is_11_l2504_250469

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def is_simplest_quadratic_radical (a : ℚ) : Prop :=
  a > 0 ∧ ¬(is_perfect_square a) ∧
  ∀ b c : ℚ, (b > 1 ∧ c > 0 ∧ a = b * c) → ¬(is_perfect_square b)

-- Theorem statement
theorem simplest_quadratic_radical_is_11 :
  is_simplest_quadratic_radical 11 ∧
  ¬(is_simplest_quadratic_radical (5/2)) ∧
  ¬(is_simplest_quadratic_radical 12) ∧
  ¬(is_simplest_quadratic_radical (1/3)) :=
sorry

end simplest_quadratic_radical_is_11_l2504_250469


namespace quadratic_roots_range_l2504_250439

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + k*x₁ - k = 0 ∧ x₂^2 + k*x₂ - k = 0) →
  (1 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3) →
  -9/2 < k ∧ k < -4 := by
sorry

end quadratic_roots_range_l2504_250439


namespace rectangle_45_odd_intersections_impossible_l2504_250436

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a grid line -/
def isOnGridLine (p : Point) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if two line segments are at 45° angle to each other -/
def isAt45Degree (p1 p2 q1 q2 : Point) : Prop :=
  |p2.x - p1.x| = |p2.y - p1.y| ∧ |q2.x - q1.x| = |q2.y - q1.y|

/-- Counts the number of grid line intersections for a line segment -/
def gridIntersections (p1 p2 : Point) : ℕ :=
  sorry

/-- Main theorem: It's impossible for all sides of a 45° rectangle to intersect an odd number of grid lines -/
theorem rectangle_45_odd_intersections_impossible (rect : Rectangle) :
  (¬ isOnGridLine rect.A) →
  (¬ isOnGridLine rect.B) →
  (¬ isOnGridLine rect.C) →
  (¬ isOnGridLine rect.D) →
  isAt45Degree rect.A rect.B rect.B rect.C →
  ¬ (Odd (gridIntersections rect.A rect.B) ∧
     Odd (gridIntersections rect.B rect.C) ∧
     Odd (gridIntersections rect.C rect.D) ∧
     Odd (gridIntersections rect.D rect.A)) :=
by sorry

end rectangle_45_odd_intersections_impossible_l2504_250436


namespace smallest_multiplier_for_perfect_square_l2504_250489

def y := 2^(3^5 * 4^4 * 5^7 * 6^5 * 7^3 * 8^6 * 9^10)

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ (∃ m : ℕ, k * y = m^2) ∧ (∀ l < k, l > 0 → ¬∃ m : ℕ, l * y = m^2) ↔ k = 70 :=
sorry

end smallest_multiplier_for_perfect_square_l2504_250489


namespace system_of_inequalities_l2504_250429

theorem system_of_inequalities (x : ℝ) :
  3 * (x + 1) > 5 * x + 4 ∧ (x - 1) / 2 ≤ (2 * x - 1) / 3 → -1 ≤ x ∧ x < -1/2 := by
  sorry

end system_of_inequalities_l2504_250429


namespace adam_ferris_wheel_cost_l2504_250444

/-- The amount Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end adam_ferris_wheel_cost_l2504_250444


namespace gemma_tip_calculation_l2504_250427

/-- Calculates the tip given to a delivery person based on the number of pizzas ordered,
    the cost per pizza, the amount paid, and the change received. -/
def calculate_tip (num_pizzas : ℕ) (cost_per_pizza : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  amount_paid - change - (num_pizzas * cost_per_pizza)

/-- Proves that given the specified conditions, the tip Gemma gave to the delivery person was $5. -/
theorem gemma_tip_calculation :
  let num_pizzas : ℕ := 4
  let cost_per_pizza : ℕ := 10
  let amount_paid : ℕ := 50
  let change : ℕ := 5
  calculate_tip num_pizzas cost_per_pizza amount_paid change = 5 := by
  sorry

#eval calculate_tip 4 10 50 5

end gemma_tip_calculation_l2504_250427


namespace pqr_product_l2504_250432

theorem pqr_product (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → 
  p + q + r = 27 → 
  1 / p + 1 / q + 1 / r + 432 / (p * q * r) = 1 → 
  p * q * r = 1380 := by
sorry

end pqr_product_l2504_250432


namespace hot_dog_remainder_l2504_250490

theorem hot_dog_remainder : 25197641 % 6 = 1 := by
  sorry

end hot_dog_remainder_l2504_250490


namespace ellipse_equation_l2504_250472

/-- Given an ellipse with the endpoint of its short axis at (3, 0) and focal distance 4,
    prove that its equation is (y²/25) + (x²/9) = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let short_axis_endpoint : ℝ × ℝ := (3, 0)
  let focal_distance : ℝ := 4
  (y^2 / 25) + (x^2 / 9) = 1 := by
sorry


end ellipse_equation_l2504_250472


namespace negation_of_existence_negation_of_quadratic_equation_l2504_250420

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l2504_250420


namespace complement_and_intersection_l2504_250435

def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}
def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem complement_and_intersection :
  (U \ A = {8, 10}) ∧ (A ∩ (U \ B) = {4, 6}) := by
  sorry

end complement_and_intersection_l2504_250435


namespace definite_integral_sin_plus_one_l2504_250450

theorem definite_integral_sin_plus_one :
  ∫ x in (-1)..(1), (Real.sin x + 1) = 2 - 2 * Real.cos 1 := by
  sorry

end definite_integral_sin_plus_one_l2504_250450


namespace termite_ridden_not_collapsing_l2504_250467

theorem termite_ridden_not_collapsing (total_homes : ℕ) 
  (termite_ridden : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden = 1 / 3 →
  collapsing_ratio = 7 / 10 →
  (termite_ridden - termite_ridden * collapsing_ratio) = 1 / 10 := by
sorry

end termite_ridden_not_collapsing_l2504_250467


namespace original_paint_intensity_l2504_250466

/-- Given a paint mixture scenario, prove that the original paint intensity was 50%. -/
theorem original_paint_intensity
  (replacement_intensity : ℝ)
  (final_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h1 : replacement_intensity = 25)
  (h2 : final_intensity = 30)
  (h3 : replaced_fraction = 0.8)
  : (1 - replaced_fraction) * 50 + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#check original_paint_intensity

end original_paint_intensity_l2504_250466


namespace banana_arrangement_count_l2504_250403

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of times 'A' appears in "BANANA" -/
def a_count : ℕ := 3

/-- The number of times 'N' appears in "BANANA" -/
def n_count : ℕ := 2

/-- The number of times 'B' appears in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count) :=
by sorry

end banana_arrangement_count_l2504_250403


namespace triangle_inequality_l2504_250484

theorem triangle_inequality (a b c r R s : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ s > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * s))
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c))) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ 
  (r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ∧
  ((1 / (a + b) + 1 / (a + c) + 1 / (b + c) = 
    r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ↔ 
   (a = b ∧ b = c)) := by
sorry

end triangle_inequality_l2504_250484


namespace hypotenuse_of_right_triangle_with_inscribed_circle_l2504_250455

/-- 
Given a right triangle with an inscribed circle, where the point of tangency 
divides one of the legs into segments of lengths m and n (m < n), 
the hypotenuse of the triangle is (m^2 + n^2) / (n - m).
-/
theorem hypotenuse_of_right_triangle_with_inscribed_circle 
  (m n : ℝ) (h : m < n) : ∃ (x : ℝ), 
  x > 0 ∧ 
  x = (m^2 + n^2) / (n - m) ∧
  x^2 = (x - n + m)^2 + (m + n)^2 := by
  sorry

end hypotenuse_of_right_triangle_with_inscribed_circle_l2504_250455


namespace total_shoes_needed_l2504_250422

def num_dogs : ℕ := 3
def num_cats : ℕ := 2
def num_ferrets : ℕ := 1
def paws_per_animal : ℕ := 4

theorem total_shoes_needed : 
  (num_dogs + num_cats + num_ferrets) * paws_per_animal = 24 := by
  sorry

end total_shoes_needed_l2504_250422


namespace billys_candy_count_l2504_250492

/-- The total number of candy pieces given the number of boxes and pieces per box -/
def total_candy (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: Billy's total candy pieces -/
theorem billys_candy_count :
  total_candy 7 3 = 21 := by
  sorry

end billys_candy_count_l2504_250492


namespace consecutive_squares_difference_l2504_250496

theorem consecutive_squares_difference (n : ℕ) : 
  (n > 0) → 
  (n + (n + 1) < 150) → 
  ((n + 1)^2 - n^2 = 129 ∨ (n + 1)^2 - n^2 = 147) :=
by
  sorry

end consecutive_squares_difference_l2504_250496


namespace apple_ratio_simplification_l2504_250416

def sarah_apples : ℕ := 630
def brother_apples : ℕ := 270
def cousin_apples : ℕ := 540

theorem apple_ratio_simplification :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    sarah_apples / k = 7 ∧ 
    brother_apples / k = 3 ∧ 
    cousin_apples / k = 6 :=
by sorry

end apple_ratio_simplification_l2504_250416


namespace probability_of_red_ball_l2504_250412

/-- Given a box with 10 balls, where 1 is yellow, 3 are green, and the rest are red,
    the probability of randomly drawing a red ball is 3/5. -/
theorem probability_of_red_ball (total_balls : ℕ) (yellow_balls : ℕ) (green_balls : ℕ) :
  total_balls = 10 →
  yellow_balls = 1 →
  green_balls = 3 →
  (total_balls - yellow_balls - green_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

#check probability_of_red_ball

end probability_of_red_ball_l2504_250412


namespace necessary_and_sufficient_condition_l2504_250477

-- Define the variables
variable (a : ℕ+) -- a is a positive integer
variable (A B : ℝ) -- A and B are real numbers
variable (x y z : ℕ+) -- x, y, z are positive integers

-- Define the system of equations
def equation1 (x y z : ℕ+) (a : ℕ+) : Prop :=
  (x : ℝ)^2 + (y : ℝ)^2 + (z : ℝ)^2 = (13 * (a : ℝ))^2

def equation2 (x y z : ℕ+) (a : ℕ+) (A B : ℝ) : Prop :=
  (x : ℝ)^2 * (A * (x : ℝ)^2 + B * (y : ℝ)^2) +
  (y : ℝ)^2 * (A * (y : ℝ)^2 + B * (z : ℝ)^2) +
  (z : ℝ)^2 * (A * (z : ℝ)^2 + B * (x : ℝ)^2) =
  1/4 * (2 * A + B) * (13 * (a : ℝ))^4

-- Theorem statement
theorem necessary_and_sufficient_condition :
  (∃ x y z : ℕ+, equation1 x y z a ∧ equation2 x y z a A B) ↔ B = 2 * A :=
sorry

end necessary_and_sufficient_condition_l2504_250477


namespace reduced_rate_fraction_is_nine_fourteenths_l2504_250434

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℚ :=
  let total_hours_per_week : ℕ := 7 * 24
  let weekday_reduced_hours : ℕ := 5 * 12
  let weekend_reduced_hours : ℕ := 2 * 24
  let total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours
  ↑total_reduced_hours / ↑total_hours_per_week

/-- Proof that the reduced rate fraction is 9/14 -/
theorem reduced_rate_fraction_is_nine_fourteenths :
  reduced_rate_fraction = 9 / 14 :=
by sorry

end reduced_rate_fraction_is_nine_fourteenths_l2504_250434


namespace january_oil_bill_l2504_250418

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 30) / january_bill = 5 / 3) →
  january_bill = 180 := by
  sorry

end january_oil_bill_l2504_250418


namespace red_cards_probability_l2504_250405

/-- The probability of drawing three red cards in succession from a deck of 60 cards,
    where 30 cards are red and 30 are black, is equal to 29/247. -/
theorem red_cards_probability (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 60) 
  (h2 : red_cards = 30) :
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 29 / 247 := by
  sorry

#eval (30 * 29 * 28) / (60 * 59 * 58)

end red_cards_probability_l2504_250405


namespace loan_principal_calculation_l2504_250423

/-- Calculates the principal amount given the interest rate, time, and total interest --/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given a loan with 12% per annum simple interest rate, 
    if the interest after 3 years is 4320, then the principal amount borrowed was 12000 --/
theorem loan_principal_calculation :
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 4320
  calculate_principal rate time interest = 12000 := by
  sorry

end loan_principal_calculation_l2504_250423


namespace set_size_comparison_l2504_250407

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end set_size_comparison_l2504_250407


namespace fifth_term_of_sequence_l2504_250424

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fifth_term_of_sequence (y : ℝ) :
  let a₁ := 3
  let r := 3 * y
  geometric_sequence a₁ r 5 = 243 * y^4 := by
  sorry

end fifth_term_of_sequence_l2504_250424


namespace consecutive_even_integers_cube_sum_l2504_250452

/-- Given three consecutive even integers whose squares sum to 2930, 
    prove that the sum of their cubes is 81720 -/
theorem consecutive_even_integers_cube_sum (n : ℤ) : 
  (∃ (n : ℤ), 
    (n^2 + (n+2)^2 + (n+4)^2 = 2930) ∧ 
    (∃ (k : ℤ), n = 2*k)) →
  n^3 + (n+2)^3 + (n+4)^3 = 81720 :=
sorry

end consecutive_even_integers_cube_sum_l2504_250452


namespace largest_root_ratio_l2504_250414

def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

def largest_root (p : ℝ → ℝ) : ℝ := sorry

theorem largest_root_ratio : 
  largest_root g / largest_root f = 2 := by sorry

end largest_root_ratio_l2504_250414


namespace product_correction_l2504_250417

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem product_correction (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 → -- a is a three-digit number
  (reverseDigits a) * b = 468 → -- incorrect calculation
  a * b = 1116 := by sorry

end product_correction_l2504_250417


namespace lcm_54_75_l2504_250438

theorem lcm_54_75 : Nat.lcm 54 75 = 1350 := by
  sorry

end lcm_54_75_l2504_250438


namespace zoom_download_time_ratio_l2504_250448

/-- Prove that the ratio of Windows download time to Mac download time is 3:1 -/
theorem zoom_download_time_ratio :
  let total_time := 82
  let mac_download_time := 10
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let glitch_time := audio_glitch_time + video_glitch_time
  let no_glitch_time := 2 * glitch_time
  let windows_download_time := total_time - (mac_download_time + glitch_time + no_glitch_time)
  (windows_download_time : ℚ) / mac_download_time = 3 / 1 := by
  sorry

end zoom_download_time_ratio_l2504_250448


namespace new_person_weight_l2504_250404

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 100 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 := by
  sorry

end new_person_weight_l2504_250404


namespace min_value_of_s_l2504_250442

theorem min_value_of_s (a b : ℤ) :
  let s := a^3 + b^3 - 60*a*b*(a + b)
  s ≥ 2012 → s ≥ 2015 :=
by sorry

end min_value_of_s_l2504_250442


namespace line_equation_l2504_250485

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

end line_equation_l2504_250485


namespace last_painted_cell_l2504_250431

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def is_painted (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

def covers_all_columns (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 → i ≤ 8 → ∃ k : ℕ, k ≤ n ∧ is_painted k ∧ k % 8 = i

theorem last_painted_cell :
  ∃ n : ℕ, n = 120 ∧ is_painted n ∧ covers_all_columns n ∧
  ∀ m : ℕ, m < n → ¬(covers_all_columns m) :=
sorry

end last_painted_cell_l2504_250431


namespace ellipse_foci_on_y_axis_l2504_250446

/-- Given that θ is an internal angle of a triangle and sin θ + cos θ = 1/2,
    prove that x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an internal angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 1/2) :
  ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    ∀ (x y : Real), 
      x^2 * Real.sin θ - y^2 * Real.cos θ = 1 ↔ 
      (x^2 / a^2) + (y^2 / b^2) = 1 ∧
      a < b :=
by sorry

end ellipse_foci_on_y_axis_l2504_250446


namespace stream_speed_l2504_250480

/-- Given a boat traveling downstream, prove the speed of the stream. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : downstream_distance = 90) 
  (h3 : downstream_time = 3) : 
  ∃ stream_speed : ℝ, 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    stream_speed = 5 := by
  sorry

end stream_speed_l2504_250480


namespace gumball_problem_l2504_250464

theorem gumball_problem (alicia_gumballs : ℕ) : 
  alicia_gumballs = 20 →
  let pedro_gumballs := alicia_gumballs + (alicia_gumballs * 3 / 2)
  let maria_gumballs := pedro_gumballs / 2
  let alicia_eaten := alicia_gumballs / 3
  let pedro_eaten := pedro_gumballs / 3
  let maria_eaten := maria_gumballs / 3
  (alicia_gumballs - alicia_eaten) + (pedro_gumballs - pedro_eaten) + (maria_gumballs - maria_eaten) = 65 := by
sorry

end gumball_problem_l2504_250464


namespace triangle_inequality_l2504_250440

theorem triangle_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2)^2 ≥ 4 * Real.tan (A / 2) * (Real.tan (A / 2) * Real.tan (C / 2) - 1) := by
  sorry

end triangle_inequality_l2504_250440


namespace equation_solution_l2504_250426

theorem equation_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 20) = (x^2 - 3*x - 18) / (x^2 - 2*x - 35) ↔ 
  x = 4 + Real.sqrt 21 ∨ x = 4 - Real.sqrt 21 :=
by sorry

end equation_solution_l2504_250426


namespace set_union_problem_l2504_250474

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end set_union_problem_l2504_250474


namespace solution_range_l2504_250491

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 5 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 7 * Real.sqrt (x - 1)) = 2 → 
  5 ≤ x ∧ x ≤ 17 := by
  sorry

end solution_range_l2504_250491


namespace congruence_problem_l2504_250449

theorem congruence_problem (c d m : ℤ) : 
  c ≡ 25 [ZMOD 53] →
  d ≡ 98 [ZMOD 53] →
  m ∈ Finset.Icc 150 200 →
  (c - d ≡ m [ZMOD 53] ↔ m = 192) := by
sorry

end congruence_problem_l2504_250449


namespace irrationality_of_cube_plus_sqrt_two_l2504_250453

theorem irrationality_of_cube_plus_sqrt_two (t : ℝ) :
  (∃ (r : ℚ), t + Real.sqrt 2 = r) → ¬ (∃ (s : ℚ), t^3 + Real.sqrt 2 = s) := by
sorry

end irrationality_of_cube_plus_sqrt_two_l2504_250453


namespace problem_statement_l2504_250415

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : b - c = 2) :
  (a - c)^2 + 3*a + 1 - 3*c = 41 := by
  sorry

end problem_statement_l2504_250415
