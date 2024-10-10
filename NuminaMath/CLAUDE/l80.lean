import Mathlib

namespace earnings_difference_l80_8067

/-- Given Paul's and Vinnie's earnings, prove the difference between them. -/
theorem earnings_difference (paul_earnings vinnie_earnings : ℕ) 
  (h1 : paul_earnings = 14)
  (h2 : vinnie_earnings = 30) : 
  vinnie_earnings - paul_earnings = 16 := by
  sorry

end earnings_difference_l80_8067


namespace solution_set_inequality_l80_8045

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (x + 1) < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end solution_set_inequality_l80_8045


namespace skier_race_l80_8069

/-- Two skiers race with given conditions -/
theorem skier_race (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (9 / y + 9 = 9 / x) ∧ (29 / y + 9 = 25 / x) → y = 1 / 4 :=
by sorry

end skier_race_l80_8069


namespace garage_wheels_count_l80_8021

/-- The number of bikes that can be assembled -/
def num_bikes : ℕ := 9

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := num_bikes * wheels_per_bike

theorem garage_wheels_count : total_wheels = 18 := by
  sorry

end garage_wheels_count_l80_8021


namespace triangle_is_obtuse_l80_8003

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(C) = b + (2/3) * c, then ABC is an obtuse triangle -/
theorem triangle_is_obtuse (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos C = b + (2/3) * c →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  π / 2 < A := by
  sorry

end triangle_is_obtuse_l80_8003


namespace evans_county_population_l80_8014

theorem evans_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 3200 →
  upper_bound = 3600 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 85000 := by
  sorry

end evans_county_population_l80_8014


namespace loss_equals_cost_of_five_balls_l80_8050

def number_of_balls : ℕ := 13
def selling_price : ℕ := 720
def cost_per_ball : ℕ := 90

theorem loss_equals_cost_of_five_balls :
  (number_of_balls * cost_per_ball - selling_price) / cost_per_ball = 5 := by
  sorry

end loss_equals_cost_of_five_balls_l80_8050


namespace two_times_binomial_seven_choose_four_l80_8081

theorem two_times_binomial_seven_choose_four : 2 * (Nat.choose 7 4) = 70 := by
  sorry

end two_times_binomial_seven_choose_four_l80_8081


namespace triangle_max_area_l80_8011

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  (1 - Real.sqrt 3 * Real.cos B) / (Real.sqrt 3 * Real.sin B) = 1 / Real.tan C →
  ∃ (S : ℝ), S = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) ∧
  ∀ (S' : ℝ), S' = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) → S' ≤ Real.sqrt 3 :=
by sorry

end triangle_max_area_l80_8011


namespace condition_equivalence_l80_8060

theorem condition_equivalence :
  (∀ x y : ℝ, x > y ↔ x^3 > y^3) ∧
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧
  (∃ x y : ℝ, x^2 > y^2 ∧ x ≤ y) :=
by sorry

end condition_equivalence_l80_8060


namespace hyperbola_properties_l80_8063

/-- The original hyperbola equation -/
def original_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

/-- The new hyperbola equation -/
def new_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 12 = 1

/-- Definition of asymptotes for a hyperbola -/
def has_same_asymptotes (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    ∀ (x y : ℝ), (f x y ↔ g (k * x) (k * y))

/-- The main theorem to prove -/
theorem hyperbola_properties :
  has_same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 2 := by sorry

end hyperbola_properties_l80_8063


namespace t_shirt_cost_l80_8059

/-- Calculates the cost of each t-shirt given the total cost, number of t-shirts and pants, and cost of each pair of pants. -/
theorem t_shirt_cost (total_cost : ℕ) (num_tshirts num_pants pants_cost : ℕ) :
  total_cost = 1500 ∧ 
  num_tshirts = 5 ∧ 
  num_pants = 4 ∧ 
  pants_cost = 250 →
  (total_cost - num_pants * pants_cost) / num_tshirts = 100 := by
  sorry

end t_shirt_cost_l80_8059


namespace smallest_h_divisible_l80_8025

theorem smallest_h_divisible : ∃! h : ℕ, 
  (∀ k : ℕ, k < h → ¬((k + 5) % 8 = 0 ∧ (k + 5) % 11 = 0 ∧ (k + 5) % 24 = 0)) ∧
  (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 :=
by sorry

end smallest_h_divisible_l80_8025


namespace roots_of_equation_l80_8084

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x - 1) = x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end roots_of_equation_l80_8084


namespace gcd_90_450_l80_8009

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by sorry

end gcd_90_450_l80_8009


namespace friends_ratio_l80_8042

theorem friends_ratio (james_friends : ℕ) (shared_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  shared_friends = 25 →
  combined_list = 275 →
  ∃ (john_friends : ℕ),
    john_friends = combined_list - james_friends →
    (john_friends : ℚ) / james_friends = 8 / 3 := by
  sorry

end friends_ratio_l80_8042


namespace square_diagonal_length_l80_8028

theorem square_diagonal_length (side_length : ℝ) (h : side_length = 30 * Real.sqrt 3) :
  Real.sqrt (2 * side_length ^ 2) = 30 * Real.sqrt 6 := by
  sorry

end square_diagonal_length_l80_8028


namespace hyperbola_x_axis_l80_8064

/-- Given k > 1, the equation (1-k)x^2 + y^2 = k^2 - 1 represents a hyperbola with its real axis along the x-axis -/
theorem hyperbola_x_axis (k : ℝ) (h : k > 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), ((1-k)*x^2 + y^2 = k^2 - 1) ↔ (x^2/a^2 - y^2/b^2 = 1) :=
sorry

end hyperbola_x_axis_l80_8064


namespace quadratic_sum_cubes_twice_product_l80_8027

theorem quadratic_sum_cubes_twice_product (m : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 6 * a + m = 0 ∧ 
              3 * b^2 + 6 * b + m = 0 ∧ 
              a ≠ b ∧ 
              a^3 + b^3 = 2 * a * b) ↔ 
  m = 6 := by
sorry

end quadratic_sum_cubes_twice_product_l80_8027


namespace melted_mixture_weight_l80_8004

def zinc_weight : ℝ := 31.5
def zinc_ratio : ℝ := 9
def copper_ratio : ℝ := 11

theorem melted_mixture_weight :
  let copper_weight := (copper_ratio / zinc_ratio) * zinc_weight
  let total_weight := zinc_weight + copper_weight
  total_weight = 70 := by sorry

end melted_mixture_weight_l80_8004


namespace roots_inside_unit_circle_iff_triangle_interior_l80_8065

/-- The region in the (a,b) plane where both roots of z^2 + az + b = 0 satisfy |z| < 1 -/
def roots_inside_unit_circle (a b : ℝ) : Prop :=
  ∀ z : ℂ, z^2 + a*z + b = 0 → Complex.abs z < 1

/-- The interior of the triangle with vertices (2, 1), (-2, 1), and (0, -1) -/
def triangle_interior (a b : ℝ) : Prop :=
  b < 1 ∧ b > a - 1 ∧ b > -a - 1 ∧ b > -1

theorem roots_inside_unit_circle_iff_triangle_interior (a b : ℝ) :
  roots_inside_unit_circle a b ↔ triangle_interior a b :=
sorry

end roots_inside_unit_circle_iff_triangle_interior_l80_8065


namespace tan_product_theorem_l80_8077

theorem tan_product_theorem (α β : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end tan_product_theorem_l80_8077


namespace equal_cost_at_60_messages_l80_8017

/-- Cost of Plan A for x text messages -/
def planACost (x : ℕ) : ℚ := 0.25 * x + 9

/-- Cost of Plan B for x text messages -/
def planBCost (x : ℕ) : ℚ := 0.40 * x

/-- The number of text messages where both plans cost the same -/
def equalCostMessages : ℕ := 60

theorem equal_cost_at_60_messages :
  planACost equalCostMessages = planBCost equalCostMessages :=
by sorry

end equal_cost_at_60_messages_l80_8017


namespace total_sessions_for_patients_l80_8090

theorem total_sessions_for_patients : 
  let num_patients : ℕ := 4
  let first_patient_sessions : ℕ := 6
  let second_patient_sessions : ℕ := first_patient_sessions + 5
  let remaining_patients_sessions : ℕ := 8
  
  num_patients = 4 →
  first_patient_sessions + 
  second_patient_sessions + 
  (num_patients - 2) * remaining_patients_sessions = 33 := by
sorry

end total_sessions_for_patients_l80_8090


namespace cookies_per_pack_l80_8040

/-- Given information about Candy's cookie distribution --/
structure CookieDistribution where
  trays : ℕ
  cookies_per_tray : ℕ
  packs : ℕ
  trays_eq : trays = 4
  cookies_per_tray_eq : cookies_per_tray = 24
  packs_eq : packs = 8

/-- Theorem: The number of cookies in each pack is 12 --/
theorem cookies_per_pack (cd : CookieDistribution) : 
  (cd.trays * cd.cookies_per_tray) / cd.packs = 12 := by
  sorry

end cookies_per_pack_l80_8040


namespace equilateral_triangle_side_length_l80_8000

theorem equilateral_triangle_side_length 
  (total_wire_length : ℝ) 
  (h1 : total_wire_length = 63) : 
  total_wire_length / 3 = 21 := by
sorry

end equilateral_triangle_side_length_l80_8000


namespace pirate_treasure_l80_8019

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end pirate_treasure_l80_8019


namespace min_sum_floor_l80_8099

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a+b+c)/d⌋ + ⌊(a+b+d)/c⌋ + ⌊(a+c+d)/b⌋ + ⌊(b+c+d)/a⌋ ≥ 8 := by
  sorry

end min_sum_floor_l80_8099


namespace wand_cost_is_60_l80_8046

/-- The cost of a magic wand at Wizards Park -/
def wand_cost : ℕ → Prop := λ x =>
  -- Kate buys 3 wands and sells 2 of them
  -- She sells each wand for $5 more than she paid
  -- She collected $130 after the sale
  2 * (x + 5) = 130

/-- The cost of each wand is $60 -/
theorem wand_cost_is_60 : wand_cost 60 := by sorry

end wand_cost_is_60_l80_8046


namespace thirteenth_divisible_by_three_l80_8088

theorem thirteenth_divisible_by_three (start : ℕ) (count : ℕ) : 
  start > 10 → 
  start % 3 = 0 → 
  ∀ n < start, n > 10 → n % 3 ≠ 0 →
  count = 13 →
  (start + 3 * (count - 1) = 48) :=
sorry

end thirteenth_divisible_by_three_l80_8088


namespace probability_different_groups_l80_8048

/-- The number of study groups -/
def num_groups : ℕ := 6

/-- The number of members in each study group -/
def members_per_group : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_groups * members_per_group

/-- The number of people to be selected -/
def selection_size : ℕ := 3

/-- The probability of selecting 3 people from different study groups -/
theorem probability_different_groups : 
  (Nat.choose num_groups selection_size : ℚ) / (Nat.choose total_people selection_size : ℚ) = 5 / 204 := by
  sorry

end probability_different_groups_l80_8048


namespace infinitely_many_solutions_l80_8062

/-- f(n) is the exponent of 2 in the prime factorization of n! -/
def f (n : ℕ+) : ℕ :=
  sorry

/-- For any positive integer a, there exist infinitely many positive integers n
    such that n - f(n) = a -/
theorem infinitely_many_solutions (a : ℕ+) :
  ∃ (S : Set ℕ+), Infinite S ∧ ∀ n ∈ S, n.val - f n = a.val := by
  sorry

end infinitely_many_solutions_l80_8062


namespace log_expression_equals_two_l80_8080

theorem log_expression_equals_two :
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + Real.log 25 = 2 := by
  sorry

end log_expression_equals_two_l80_8080


namespace flower_purchase_solution_l80_8082

/-- Represents the flower purchase problem -/
structure FlowerPurchase where
  priceA : ℕ  -- Price of type A flower
  priceB : ℕ  -- Price of type B flower
  totalPlants : ℕ  -- Total number of plants to purchase
  (first_purchase : 30 * priceA + 15 * priceB = 675)
  (second_purchase : 12 * priceA + 5 * priceB = 265)
  (total_constraint : totalPlants = 31)
  (type_b_constraint : ∀ m : ℕ, m ≤ totalPlants → totalPlants - m < 2 * m)

/-- Theorem stating the solution to the flower purchase problem -/
theorem flower_purchase_solution (fp : FlowerPurchase) :
  fp.priceA = 20 ∧ fp.priceB = 5 ∧
  ∃ (m : ℕ), m = 11 ∧ fp.totalPlants - m = 20 ∧
  20 * m + 5 * (fp.totalPlants - m) = 320 ∧
  ∀ (n : ℕ), n ≤ fp.totalPlants → 
    20 * n + 5 * (fp.totalPlants - n) ≥ 20 * m + 5 * (fp.totalPlants - m) :=
by sorry


end flower_purchase_solution_l80_8082


namespace greg_savings_l80_8095

theorem greg_savings (scooter_cost : ℕ) (amount_needed : ℕ) (amount_saved : ℕ) : 
  scooter_cost = 90 → amount_needed = 33 → amount_saved = scooter_cost - amount_needed → amount_saved = 57 := by
sorry

end greg_savings_l80_8095


namespace average_of_remaining_numbers_l80_8092

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset : ℕ)
  (avg_all : ℚ)
  (avg_subset : ℚ)
  (h_total : n = 5)
  (h_subset : subset = 3)
  (h_avg_all : avg_all = 6)
  (h_avg_subset : avg_subset = 4) :
  (n * avg_all - subset * avg_subset) / (n - subset) = 9 := by
sorry

end average_of_remaining_numbers_l80_8092


namespace triangle_is_equilateral_l80_8036

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
  (h2 : t.b^2 = t.a * t.c)      -- b is geometric mean of a and c
  : t.A = t.B ∧ t.B = t.C ∧ t.a = t.b ∧ t.b = t.c :=
by sorry

end triangle_is_equilateral_l80_8036


namespace smallest_x_absolute_value_equation_l80_8073

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x = -5.5 ∧ |4*x + 7| = 15 ∧ ∀ (y : ℝ), |4*y + 7| = 15 → y ≥ x :=
by sorry

end smallest_x_absolute_value_equation_l80_8073


namespace telephone_network_connections_l80_8097

/-- The number of distinct connections in a network of telephones -/
def distinct_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 7 telephones, where each telephone is connected to 6 others,
    the total number of distinct connections is 21. -/
theorem telephone_network_connections :
  distinct_connections 7 6 = 21 := by
  sorry

end telephone_network_connections_l80_8097


namespace subtraction_problem_l80_8039

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end subtraction_problem_l80_8039


namespace b_value_l80_8020

theorem b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end b_value_l80_8020


namespace river_current_speed_l80_8012

/-- Theorem: Given a swimmer's speed in still water and the ratio of upstream to downstream swimming time, we can determine the speed of the river's current. -/
theorem river_current_speed 
  (swimmer_speed : ℝ) 
  (upstream_downstream_ratio : ℝ) 
  (h1 : swimmer_speed = 10) 
  (h2 : upstream_downstream_ratio = 3) : 
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
  (swimmer_speed + current_speed) * upstream_downstream_ratio = 
  (swimmer_speed - current_speed) * (upstream_downstream_ratio + 1) := by
  sorry

end river_current_speed_l80_8012


namespace longest_chord_implies_a_equals_one_l80_8091

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The circle equation (x-1)^2 + (y-a)^2 = 4 -/
def circle_equation (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4

/-- A point (x, y) is on the circle -/
def point_on_circle (a x y : ℝ) : Prop := circle_equation a x y

/-- A point (x, y) is on the line -/
def point_on_line (a x y : ℝ) : Prop := line_equation a x y

/-- The theorem to be proved -/
theorem longest_chord_implies_a_equals_one (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle a x₁ y₁ ∧
    point_on_circle a x₂ y₂ ∧
    point_on_line a x₁ y₁ ∧
    point_on_line a x₂ y₂ ∧
    ∀ x y : ℝ, point_on_circle a x y → (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (x - x₁)^2 + (y - y₁)^2) →
  a = 1 := by sorry

end longest_chord_implies_a_equals_one_l80_8091


namespace f_of_one_equals_twentyone_l80_8022

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 3 * (x + 3) + 1

-- State the theorem
theorem f_of_one_equals_twentyone : f 1 = 21 := by sorry

end f_of_one_equals_twentyone_l80_8022


namespace probability_sum_6_is_5_36_l80_8018

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of combinations that result in a sum of 6 -/
def favorable_outcomes : ℕ := 5

/-- The probability of rolling a sum of 6 with two dice -/
def probability_sum_6 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_6_is_5_36 : probability_sum_6 = 5 / 36 := by
  sorry

end probability_sum_6_is_5_36_l80_8018


namespace correct_option_is_B_l80_8037

-- Define the statements
def statement1 : Prop := False
def statement2 : Prop := True
def statement3 : Prop := True
def statement4 : Prop := False

-- Define the options
def optionA : Prop := statement1 ∧ statement2 ∧ statement3
def optionB : Prop := statement2 ∧ statement3
def optionC : Prop := statement2 ∧ statement4
def optionD : Prop := statement1 ∧ statement3 ∧ statement4

-- Theorem: The correct option is B
theorem correct_option_is_B : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) → 
  (optionB ∧ ¬optionA ∧ ¬optionC ∧ ¬optionD) :=
by sorry

end correct_option_is_B_l80_8037


namespace min_time_to_finish_tasks_l80_8030

def wash_rice_time : ℕ := 2
def cook_porridge_time : ℕ := 10
def wash_vegetables_time : ℕ := 3
def chop_vegetables_time : ℕ := 5

def total_vegetable_time : ℕ := wash_vegetables_time + chop_vegetables_time

theorem min_time_to_finish_tasks : ℕ := by
  have h1 : wash_rice_time + cook_porridge_time = 12 := by sorry
  have h2 : total_vegetable_time ≤ cook_porridge_time := by sorry
  exact 12

end min_time_to_finish_tasks_l80_8030


namespace solution_set_implies_a_value_l80_8085

/-- If the solution set of the inequality |ax+2| < 6 is (-1,2), then a = -4 -/
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a*x + 2| < 6 ↔ -1 < x ∧ x < 2) →
  a = -4 := by
sorry

end solution_set_implies_a_value_l80_8085


namespace genevieve_money_proof_l80_8033

/-- The amount of money Genevieve had initially -/
def genevieve_initial_amount (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ) : ℚ :=
  cost_per_kg * bought_kg - short_amount

/-- Proof that Genevieve's initial amount was $1600 -/
theorem genevieve_money_proof (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ)
  (h1 : cost_per_kg = 8)
  (h2 : bought_kg = 250)
  (h3 : short_amount = 400) :
  genevieve_initial_amount cost_per_kg bought_kg short_amount = 1600 := by
  sorry

end genevieve_money_proof_l80_8033


namespace dog_paws_on_ground_l80_8008

theorem dog_paws_on_ground : ∀ (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_fours : ℕ),
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_fours = total_dogs / 2 →
  dogs_on_back_legs + dogs_on_all_fours = total_dogs →
  dogs_on_all_fours * 4 + dogs_on_back_legs * 2 = 36 := by
  sorry

end dog_paws_on_ground_l80_8008


namespace least_value_quadratic_equation_l80_8058

theorem least_value_quadratic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^2 + 5 * y + 2
  ∃ y_min : ℝ, (f y_min = 4) ∧ (∀ y : ℝ, f y = 4 → y ≥ y_min) ∧ y_min = -2 := by
  sorry

end least_value_quadratic_equation_l80_8058


namespace range_of_f_l80_8079

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem range_of_f :
  Set.range f = Set.Ioo 0 3 := by sorry

end range_of_f_l80_8079


namespace opposite_direction_speed_l80_8031

/-- Given two people moving in opposite directions, with one person's speed and their final separation known, prove the other person's speed. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (total_separation : ℝ) 
  (time : ℝ) 
  (h1 : riya_speed = 21)
  (h2 : total_separation = 43)
  (h3 : time = 1) :
  let priya_speed := total_separation / time - riya_speed
  priya_speed = 22 := by sorry

end opposite_direction_speed_l80_8031


namespace polynomial_coefficient_sum_l80_8015

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 32 := by
  sorry

end polynomial_coefficient_sum_l80_8015


namespace sum_of_digits_l80_8089

theorem sum_of_digits (A B C D E : ℕ) : A + B + C + D + E = 32 :=
  by
  have h1 : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 := by sorry
  have h2 : 3 * E % 10 = 1 := by sorry
  have h3 : 3 * A + (B + C + D + 2) = 20 := by sorry
  have h4 : B + C + D = 19 := by sorry
  sorry

end sum_of_digits_l80_8089


namespace triangle_vertices_from_midpoints_l80_8049

/-- Given a triangle with midpoints, prove its vertices -/
theorem triangle_vertices_from_midpoints :
  let m1 : ℚ × ℚ := (1/4, 13/4)
  let m2 : ℚ × ℚ := (-1/2, 1)
  let m3 : ℚ × ℚ := (-5/4, 5/4)
  let v1 : ℚ × ℚ := (-2, -1)
  let v2 : ℚ × ℚ := (-1/2, 13/4)
  let v3 : ℚ × ℚ := (1, 7/2)
  (m1.1 = (v2.1 + v3.1) / 2 ∧ m1.2 = (v2.2 + v3.2) / 2) ∧
  (m2.1 = (v1.1 + v3.1) / 2 ∧ m2.2 = (v1.2 + v3.2) / 2) ∧
  (m3.1 = (v1.1 + v2.1) / 2 ∧ m3.2 = (v1.2 + v2.2) / 2) :=
by
  sorry


end triangle_vertices_from_midpoints_l80_8049


namespace fraction_addition_l80_8038

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l80_8038


namespace tripled_base_and_exponent_l80_8010

theorem tripled_base_and_exponent (c d : ℤ) (y : ℚ) (h1 : d ≠ 0) :
  (3 * c : ℚ) ^ (3 * d) = c ^ d * y ^ d → y = 27 * c ^ 2 := by
  sorry

end tripled_base_and_exponent_l80_8010


namespace fraction_multiplication_l80_8006

theorem fraction_multiplication :
  (2 : ℚ) / 3 * 5 / 7 * 9 / 13 * 4 / 11 = 120 / 1001 := by
  sorry

end fraction_multiplication_l80_8006


namespace hcf_of_three_numbers_l80_8002

theorem hcf_of_three_numbers (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a.val b.val) c.val = 1200) →
  (a.val * b.val * c.val = 108000) →
  (Nat.gcd (Nat.gcd a.val b.val) c.val = 90) := by
sorry

end hcf_of_three_numbers_l80_8002


namespace equilateral_triangle_not_unique_l80_8016

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  angle : ℝ

/-- Given one angle and the side opposite to it, an equilateral triangle is not uniquely determined -/
theorem equilateral_triangle_not_unique (α : ℝ) (s : ℝ) : 
  ∃ (t1 t2 : EquilateralTriangle), t1.angle = α ∧ t1.side = s ∧ t2.angle = α ∧ t2.side = s ∧ t1 ≠ t2 :=
sorry

end equilateral_triangle_not_unique_l80_8016


namespace five_dice_not_same_probability_l80_8043

theorem five_dice_not_same_probability :
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 5  -- number of dice rolled
  let total_outcomes : ℕ := n^k
  let same_number_outcomes : ℕ := n
  let not_same_number_probability : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  not_same_number_probability = 1295 / 1296 :=
by sorry

end five_dice_not_same_probability_l80_8043


namespace round_trip_fuel_efficiency_l80_8066

/-- Calculates the average fuel efficiency for a round trip given the conditions. -/
theorem round_trip_fuel_efficiency 
  (distance : ℝ) 
  (efficiency1 : ℝ) 
  (efficiency2 : ℝ) 
  (h1 : distance = 120) 
  (h2 : efficiency1 = 30) 
  (h3 : efficiency2 = 20) : 
  (2 * distance) / (distance / efficiency1 + distance / efficiency2) = 24 :=
by
  sorry

#check round_trip_fuel_efficiency

end round_trip_fuel_efficiency_l80_8066


namespace sasha_muffins_count_l80_8051

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 50

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 4 * sasha_muffins

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

/-- The price of each muffin in cents -/
def muffin_price : ℕ := 400

/-- The total amount raised in cents -/
def total_raised : ℕ := 90000

theorem sasha_muffins_count : 
  sasha_muffins = 50 ∧ 
  melissa_muffins = 4 * sasha_muffins ∧
  tiffany_muffins = (sasha_muffins + melissa_muffins) / 2 ∧
  total_muffins * muffin_price = total_raised := by
  sorry

end sasha_muffins_count_l80_8051


namespace max_sector_area_l80_8026

/-- The maximum area of a sector with circumference 4 -/
theorem max_sector_area (r l : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : 2*r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end max_sector_area_l80_8026


namespace decimal_85_equals_base7_151_l80_8007

/-- Converts a number from decimal to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to decimal --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_85_equals_base7_151 : fromBase7 [1, 5, 1] = 85 := by
  sorry

#eval toBase7 85  -- Should output [1, 5, 1]
#eval fromBase7 [1, 5, 1]  -- Should output 85

end decimal_85_equals_base7_151_l80_8007


namespace cat_resisting_time_l80_8034

/-- Proves that given a total time of 28 minutes, a walking distance of 64 feet,
    and a walking rate of 8 feet/minute, the time spent resisting is 20 minutes. -/
theorem cat_resisting_time
  (total_time : ℕ)
  (walking_distance : ℕ)
  (walking_rate : ℕ)
  (h1 : total_time = 28)
  (h2 : walking_distance = 64)
  (h3 : walking_rate = 8)
  : total_time - walking_distance / walking_rate = 20 := by
  sorry

#check cat_resisting_time

end cat_resisting_time_l80_8034


namespace two_integer_k_values_for_nontrivial_solution_l80_8057

/-- The system of equations has a non-trivial solution for exactly two integer values of k. -/
theorem two_integer_k_values_for_nontrivial_solution :
  ∃! (s : Finset ℤ), (∀ k ∈ s, ∃ a b c : ℝ, (a, b, c) ≠ (0, 0, 0) ∧
    a^2 + b^2 = k * c * (a + b) ∧
    b^2 + c^2 = k * a * (b + c) ∧
    c^2 + a^2 = k * b * (c + a)) ∧
  s.card = 2 := by
  sorry

end two_integer_k_values_for_nontrivial_solution_l80_8057


namespace distribute_balls_count_l80_8061

/-- The number of ways to put 4 different balls into 4 different boxes, leaving exactly two boxes empty -/
def ways_to_distribute_balls : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of ways to distribute the balls is 84 -/
theorem distribute_balls_count : ways_to_distribute_balls = 84 := by
  sorry

end distribute_balls_count_l80_8061


namespace percentage_of_difference_l80_8094

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (40 / 100) * (x + y) →
  y = (11.11111111111111 / 100) * x →
  P = 6.25 := by
sorry

end percentage_of_difference_l80_8094


namespace max_pairs_with_distinct_sums_l80_8044

/-- Given a set of integers from 1 to 2010, we can choose at most 803 pairs
    such that the elements of each pair are distinct, no two pairs share an element,
    and the sum of each pair is unique and not greater than 2010. -/
theorem max_pairs_with_distinct_sums :
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2010) ∧
    pairs.card = k ∧
    (∀ (m : ℕ) (other_pairs : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 + p.2 ≤ 2010) →
      other_pairs.card = m →
      m ≤ k) :=
by sorry

end max_pairs_with_distinct_sums_l80_8044


namespace square_of_complex_number_l80_8087

theorem square_of_complex_number :
  let z : ℂ := 5 - 2 * Complex.I
  z * z = 21 - 20 * Complex.I :=
by sorry

end square_of_complex_number_l80_8087


namespace inverse_proportion_values_l80_8072

/-- α is inversely proportional to β with α = 5 when β = -4 -/
def inverse_proportion (α β : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ α * β = k ∧ 5 * (-4) = k

theorem inverse_proportion_values (α β : ℝ) (h : inverse_proportion α β) :
  (β = -10 → α = 2) ∧ (β = 2 → α = -10) := by sorry

end inverse_proportion_values_l80_8072


namespace square_ratio_theorem_l80_8074

theorem square_ratio_theorem (area_ratio : ℚ) (side_ratio : ℚ) 
  (a b c : ℕ) (h1 : area_ratio = 50 / 98) :
  side_ratio = Real.sqrt (area_ratio) ∧
  side_ratio = 5 / 7 ∧
  (a : ℚ) * Real.sqrt b / (c : ℚ) = side_ratio ∧
  a + b + c = 12 := by
  sorry

end square_ratio_theorem_l80_8074


namespace may_savings_l80_8056

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end may_savings_l80_8056


namespace functional_equation_solution_l80_8093

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that the only functions satisfying the equation are f(x) = 0 or f(x) = x² -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x^2) := by
  sorry

end functional_equation_solution_l80_8093


namespace equation_solution_l80_8078

theorem equation_solution (x : ℝ) : 
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) - 1 / (x + 8) - 1 / (x + 10) + 1 / (x + 12) + 1 / (x + 14) = 0) ↔ 
  (x = -7 ∨ x = -7 + Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 + Real.sqrt (19 - 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 - 6 * Real.sqrt 5)) :=
by sorry

end equation_solution_l80_8078


namespace negation_of_existence_is_universal_nonequality_l80_8083

theorem negation_of_existence_is_universal_nonequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end negation_of_existence_is_universal_nonequality_l80_8083


namespace eggs_per_basket_l80_8086

theorem eggs_per_basket : ∀ (n : ℕ),
  (30 % n = 0) →  -- Yellow eggs are evenly distributed
  (42 % n = 0) →  -- Blue eggs are evenly distributed
  (n ≥ 4) →       -- At least 4 eggs per basket
  (30 / n ≥ 3) →  -- At least 3 purple baskets
  (42 / n ≥ 3) →  -- At least 3 orange baskets
  n = 6 :=
by
  sorry

#check eggs_per_basket

end eggs_per_basket_l80_8086


namespace mobile_phone_sales_growth_l80_8047

/-- Represents the sales growth of mobile phones over two months -/
theorem mobile_phone_sales_growth 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (monthly_growth_rate : ℝ) 
  (h1 : initial_sales = 400) 
  (h2 : final_sales = 900) :
  initial_sales * (1 + monthly_growth_rate)^2 = final_sales := by
  sorry

end mobile_phone_sales_growth_l80_8047


namespace rotation_of_A_about_B_l80_8023

-- Define the points
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)

-- Define the rotation function
def rotate180AboutPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (cx, cy) := center
  (2 * cx - px, 2 * cy - py)

-- Theorem statement
theorem rotation_of_A_about_B :
  rotate180AboutPoint A B = (2, 7) := by sorry

end rotation_of_A_about_B_l80_8023


namespace subset_implies_a_greater_than_half_l80_8029

-- Define the sets M and N
def M : Set ℝ := {x | -2 * x + 1 ≥ 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_greater_than_half (a : ℝ) :
  M ⊆ N a → a > 1/2 := by
  sorry

end subset_implies_a_greater_than_half_l80_8029


namespace area_triangle_DEF_l80_8001

/-- Triangle DEF with hypotenuse DE, angle between DF and DE is 45°, and length of DF is 4 units -/
structure Triangle_DEF where
  DE : ℝ  -- Length of hypotenuse DE
  DF : ℝ  -- Length of side DF
  EF : ℝ  -- Length of side EF
  angle_DF_DE : ℝ  -- Angle between DF and DE in radians
  hypotenuse_DE : DE = DF * Real.sqrt 2  -- DE is hypotenuse
  angle_45_deg : angle_DF_DE = π / 4  -- Angle is 45°
  DF_length : DF = 4  -- Length of DF is 4 units

/-- The area of triangle DEF is 8 square units -/
theorem area_triangle_DEF (t : Triangle_DEF) : (1 / 2) * t.DF * t.EF = 8 := by
  sorry

end area_triangle_DEF_l80_8001


namespace complex_equation_proof_l80_8070

theorem complex_equation_proof (a b : ℝ) : (a - 2 * I) / I = b + I → a - b = 1 := by
  sorry

end complex_equation_proof_l80_8070


namespace inductive_reasoning_correct_l80_8096

-- Define the types of reasoning
inductive ReasoningMethod
| Analogical
| Deductive
| Inductive
| Reasonable

-- Define the direction of reasoning
inductive ReasoningDirection
| IndividualToIndividual
| GeneralToSpecific
| IndividualToGeneral
| Other

-- Define a function that describes the direction of each reasoning method
def reasoningDirection (method : ReasoningMethod) : ReasoningDirection :=
  match method with
  | ReasoningMethod.Analogical => ReasoningDirection.IndividualToIndividual
  | ReasoningMethod.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningMethod.Inductive => ReasoningDirection.IndividualToGeneral
  | ReasoningMethod.Reasonable => ReasoningDirection.Other

-- Define a predicate for whether a reasoning method can be used in a proof
def canBeUsedInProof (method : ReasoningMethod) : Prop :=
  match method with
  | ReasoningMethod.Reasonable => False
  | _ => True

-- Theorem stating that inductive reasoning is the only correct answer
theorem inductive_reasoning_correct :
  (∀ m : ReasoningMethod, m ≠ ReasoningMethod.Inductive →
    (reasoningDirection m ≠ ReasoningDirection.IndividualToGeneral ∨
     ¬canBeUsedInProof m)) ∧
  (reasoningDirection ReasoningMethod.Inductive = ReasoningDirection.IndividualToGeneral ∧
   canBeUsedInProof ReasoningMethod.Inductive) :=
by
  sorry


end inductive_reasoning_correct_l80_8096


namespace annual_piano_clarinet_cost_difference_l80_8098

/-- Calculates the difference in annual cost between piano and clarinet lessons --/
def annual_lesson_cost_difference (clarinet_hourly_rate piano_hourly_rate : ℕ) 
  (clarinet_weekly_hours piano_weekly_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  ((piano_hourly_rate * piano_weekly_hours) - (clarinet_hourly_rate * clarinet_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual cost between piano and clarinet lessons is $1040 --/
theorem annual_piano_clarinet_cost_difference : 
  annual_lesson_cost_difference 40 28 3 5 52 = 1040 := by
  sorry

end annual_piano_clarinet_cost_difference_l80_8098


namespace absolute_value_equation_solution_l80_8076

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 30| + |x - 20| = |3*x - 90| :=
by
  -- The unique solution is x = 40
  use 40
  sorry

end absolute_value_equation_solution_l80_8076


namespace yogurt_combinations_yogurt_shop_combinations_l80_8053

theorem yogurt_combinations (n : ℕ) (k : ℕ) : n ≥ k → (n.choose k) = n.factorial / (k.factorial * (n - k).factorial) := by sorry

theorem yogurt_shop_combinations : 
  (5 : ℕ) * ((7 : ℕ).choose 3) = 175 := by sorry

end yogurt_combinations_yogurt_shop_combinations_l80_8053


namespace cube_volume_problem_l80_8005

theorem cube_volume_problem (a : ℕ) : 
  (a + 1) * (a + 1) * (a - 2) = a^3 - 27 → a^3 = 125 := by
  sorry

end cube_volume_problem_l80_8005


namespace jake_newspaper_count_l80_8041

/-- The number of newspapers Jake delivers in a week -/
def jake_newspapers : ℕ := 234

/-- The number of newspapers Miranda delivers in a week -/
def miranda_newspapers : ℕ := 2 * jake_newspapers

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

theorem jake_newspaper_count : jake_newspapers = 234 :=
  by
    have h1 : miranda_newspapers = 2 * jake_newspapers := by rfl
    have h2 : weeks_in_month * miranda_newspapers - weeks_in_month * jake_newspapers = 936 :=
      by sorry
    sorry

end jake_newspaper_count_l80_8041


namespace line_equation_proof_l80_8068

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equation_proof (l : Line) :
  l.contains (0, 3) ∧
  l.perpendicular ⟨1, 1, 1⟩ →
  l = ⟨1, -1, 3⟩ := by
  sorry

#check line_equation_proof

end line_equation_proof_l80_8068


namespace arithmetic_mean_difference_l80_8024

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end arithmetic_mean_difference_l80_8024


namespace fraction_zero_iff_x_plus_minus_five_l80_8035

theorem fraction_zero_iff_x_plus_minus_five (x : ℝ) :
  (x^2 - 25) / (4 * x^2 - 2 * x) = 0 ↔ x = 5 ∨ x = -5 :=
by sorry

end fraction_zero_iff_x_plus_minus_five_l80_8035


namespace baker_pastries_l80_8054

theorem baker_pastries (cakes_made : ℕ) (pastries_sold : ℕ) (total_cakes_sold : ℕ) (difference : ℕ) :
  cakes_made = 14 →
  pastries_sold = 8 →
  total_cakes_sold = 97 →
  total_cakes_sold - pastries_sold = difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end baker_pastries_l80_8054


namespace french_toast_loaves_l80_8055

/-- Calculates the number of loaves of bread needed for french toast over a given number of weeks -/
def loaves_needed (slices_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (slices_per_loaf : ℕ) : ℕ :=
  (slices_per_day * days_per_week * weeks + slices_per_loaf - 1) / slices_per_loaf

theorem french_toast_loaves :
  let slices_per_day : ℕ := 3  -- Suzanne (1) + husband (1) + daughters (0.5 + 0.5)
  let days_per_week : ℕ := 2   -- Saturday and Sunday
  let weeks : ℕ := 52
  let slices_per_loaf : ℕ := 12
  loaves_needed slices_per_day days_per_week weeks slices_per_loaf = 26 := by
  sorry

#eval loaves_needed 3 2 52 12

end french_toast_loaves_l80_8055


namespace nabla_problem_l80_8032

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^(a - 1)

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 1027 := by
  sorry

end nabla_problem_l80_8032


namespace ticket_cost_theorem_l80_8071

def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10
def sister_discount : ℝ := 0.30

def num_adults : ℕ := 5
def num_children : ℕ := 4
def num_seniors : ℕ := 3

def total_cost : ℝ :=
  (adult_price * (1 - husband_discount) + adult_price) +  -- Mrs. Lopez and husband
  (senior_price * 2 * (1 - parents_discount)) +           -- Parents
  (child_price * 3 + child_price + adult_price * (1 - nephew_discount)) + -- Children and nephews
  senior_price +                                          -- Aunt (buy-one-get-one-free)
  (adult_price * 2) +                                     -- Two friends
  (adult_price * (1 - sister_discount))                   -- Sister

theorem ticket_cost_theorem : total_cost = 115.15 := by
  sorry

end ticket_cost_theorem_l80_8071


namespace complex_trajectory_l80_8013

theorem complex_trajectory (x y : ℝ) (h1 : x ≥ (1/2 : ℝ)) (z : ℂ) 
  (h2 : z = Complex.mk x y) (h3 : Complex.abs (z - 1) = x) : 
  y^2 = 2*x - 1 := by
sorry

end complex_trajectory_l80_8013


namespace geometric_sequence_increasing_condition_l80_8075

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem stating that "q > 1" is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(GeometricSequence a q ∧ (q > 1 ↔ MonotonicallyIncreasing a)) :=
sorry

end geometric_sequence_increasing_condition_l80_8075


namespace employed_males_percentage_l80_8052

theorem employed_males_percentage (total_employed_percent : Real) 
  (employed_females_percent : Real) (h1 : total_employed_percent = 64) 
  (h2 : employed_females_percent = 28.125) : 
  (total_employed_percent / 100) * (100 - employed_females_percent) = 45.96 :=
sorry

end employed_males_percentage_l80_8052
