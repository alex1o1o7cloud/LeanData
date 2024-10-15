import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l45_4582

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 33 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l45_4582


namespace NUMINAMATH_CALUDE_square_difference_given_product_and_sum_l45_4539

theorem square_difference_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_product_and_sum_l45_4539


namespace NUMINAMATH_CALUDE_sum_is_positive_difference_is_negative_four_l45_4564

variables (a b : ℝ)

def A : ℝ := a^2 - 2*a*b + b^2
def B : ℝ := a^2 + 2*a*b + b^2

theorem sum_is_positive (h : a ≠ b) : A a b + B a b > 0 := by
  sorry

theorem difference_is_negative_four (h : a * b = 1) : A a b - B a b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_positive_difference_is_negative_four_l45_4564


namespace NUMINAMATH_CALUDE_problem_solution_l45_4513

-- Define the functions f and g
def f (a b x : ℝ) := |x - a| - |x + b|
def g (a b x : ℝ) := -x^2 - a*x - b

-- State the theorem
theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hf_max : ∃ x, f a b x = 3) : 
  (a + b = 3) ∧ 
  (∀ x ≥ a, g a b x < f a b x) → 
  (1/2 < a ∧ a < 3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l45_4513


namespace NUMINAMATH_CALUDE_exists_society_with_subgroup_l45_4575

/-- Definition of a society with n girls and m boys -/
structure Society :=
  (n : ℕ) -- number of girls
  (m : ℕ) -- number of boys

/-- Definition of a relationship between boys and girls in a society -/
def Knows (s : Society) := 
  Fin s.m → Fin s.n → Prop

/-- Definition of a subgroup with the required property -/
def HasSubgroup (s : Society) (knows : Knows s) : Prop :=
  ∃ (girls : Fin 5 → Fin s.n) (boys : Fin 5 → Fin s.m),
    (∀ i j, knows (boys i) (girls j)) ∨ 
    (∀ i j, ¬knows (boys i) (girls j))

/-- Main theorem: Existence of n₀ and m₀ satisfying the property -/
theorem exists_society_with_subgroup :
  ∃ (n₀ m₀ : ℕ), ∀ (s : Society),
    s.n = n₀ → s.m = m₀ → 
    ∀ (knows : Knows s), HasSubgroup s knows :=
sorry

end NUMINAMATH_CALUDE_exists_society_with_subgroup_l45_4575


namespace NUMINAMATH_CALUDE_rectangle_area_l45_4535

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l45_4535


namespace NUMINAMATH_CALUDE_M_union_S_eq_M_l45_4571

-- Define set M
def M : Set ℝ := {y | ∃ x, y = Real.exp (x * Real.log 2)}

-- Define set S
def S : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem M_union_S_eq_M : M ∪ S = M := by
  sorry

end NUMINAMATH_CALUDE_M_union_S_eq_M_l45_4571


namespace NUMINAMATH_CALUDE_regular_octagon_area_l45_4514

theorem regular_octagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let square_side : ℝ := 2 + s
  let octagon_area : ℝ := square_side ^ 2 - 4 * (1 / 2)
  octagon_area = 4 + 4 * s := by sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l45_4514


namespace NUMINAMATH_CALUDE_rice_cost_problem_l45_4512

/-- Proves that the cost of the first type of rice is 16 rupees per kg -/
theorem rice_cost_problem (rice1_weight : ℝ) (rice2_weight : ℝ) (rice2_cost : ℝ) (avg_cost : ℝ) 
  (h1 : rice1_weight = 8)
  (h2 : rice2_weight = 4)
  (h3 : rice2_cost = 22)
  (h4 : avg_cost = 18)
  (h5 : (rice1_weight * rice1_cost + rice2_weight * rice2_cost) / (rice1_weight + rice2_weight) = avg_cost) :
  rice1_cost = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rice_cost_problem_l45_4512


namespace NUMINAMATH_CALUDE_allan_brought_two_balloons_l45_4507

/-- The number of balloons Allan and Jake had in total -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := 4

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := total_balloons - jake_balloons

theorem allan_brought_two_balloons : allan_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_brought_two_balloons_l45_4507


namespace NUMINAMATH_CALUDE_popcorn_cost_l45_4581

/-- The cost of each box of popcorn for three friends splitting movie expenses -/
theorem popcorn_cost (ticket_price movie_tickets popcorn_boxes milktea_price milktea_cups individual_contribution : ℚ) :
  (ticket_price = 7) →
  (movie_tickets = 3) →
  (popcorn_boxes = 2) →
  (milktea_price = 3) →
  (milktea_cups = 3) →
  (individual_contribution = 11) →
  (((ticket_price * movie_tickets) + (milktea_price * milktea_cups) + 
    (popcorn_boxes * ((individual_contribution * 3) - 
    (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes)) / 3 = individual_contribution) →
  ((individual_contribution * 3) - (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes = (3/2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_popcorn_cost_l45_4581


namespace NUMINAMATH_CALUDE_salmon_sales_ratio_l45_4586

/-- Given the first week's salmon sales and the total sales over two weeks,
    prove that the ratio of the second week's sales to the first week's sales is 3:1 -/
theorem salmon_sales_ratio (first_week : ℝ) (total : ℝ) :
  first_week = 50 →
  total = 200 →
  (total - first_week) / first_week = 3 := by
sorry

end NUMINAMATH_CALUDE_salmon_sales_ratio_l45_4586


namespace NUMINAMATH_CALUDE_ratio_of_balls_l45_4584

def red_balls : ℕ := 16
def white_balls : ℕ := 20

theorem ratio_of_balls : 
  (red_balls : ℚ) / white_balls = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_balls_l45_4584


namespace NUMINAMATH_CALUDE_remainder_fraction_l45_4598

theorem remainder_fraction (x : ℝ) (h : x = 62.5) : 
  ((x + 5) * 2 / 5 - 5) / 44 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_remainder_fraction_l45_4598


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l45_4559

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 40)
  (lcm_eq : Nat.lcm a b = 40 * 11 * 12) :
  max a b = 480 := by
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l45_4559


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l45_4592

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_percentage : ℝ := 0.40

-- Define the percentage of employees who have fair hair
def fair_haired_percentage : ℝ := 0.70

-- Theorem statement
theorem women_fair_hair_percentage :
  fair_haired_women_percentage * fair_haired_percentage = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l45_4592


namespace NUMINAMATH_CALUDE_profit_distribution_l45_4563

/-- Represents the profit distribution problem -/
theorem profit_distribution 
  (john_investment : ℕ) (john_months : ℕ)
  (rose_investment : ℕ) (rose_months : ℕ)
  (tom_investment : ℕ) (tom_months : ℕ)
  (profit_share_diff : ℕ) :
  john_investment = 18000 →
  john_months = 12 →
  rose_investment = 12000 →
  rose_months = 9 →
  tom_investment = 9000 →
  tom_months = 8 →
  profit_share_diff = 370 →
  ∃ (total_profit : ℕ),
    total_profit = 4070 ∧
    (rose_investment * rose_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) -
    (tom_investment * tom_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) = 
    profit_share_diff :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_l45_4563


namespace NUMINAMATH_CALUDE_coordinates_of_G_l45_4505

/-- Given a line segment OH with O at (0, 0) and H at (12, 0), 
    and a point G on the same vertical line as H,
    if the line from G through the midpoint M of OH intersects the y-axis at P(0, -4),
    then G has coordinates (12, 4) -/
theorem coordinates_of_G (O H G M P : ℝ × ℝ) : 
  O = (0, 0) →
  H = (12, 0) →
  G.1 = H.1 →
  M = ((O.1 + H.1) / 2, (O.2 + H.2) / 2) →
  P = (0, -4) →
  (∃ t : ℝ, G = t • (M - P) + P) →
  G = (12, 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_G_l45_4505


namespace NUMINAMATH_CALUDE_dave_new_cards_l45_4500

/-- Calculates the number of new baseball cards given the total pages used,
    cards per page, and number of old cards. -/
def new_cards (pages : ℕ) (cards_per_page : ℕ) (old_cards : ℕ) : ℕ :=
  pages * cards_per_page - old_cards

theorem dave_new_cards :
  new_cards 2 8 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dave_new_cards_l45_4500


namespace NUMINAMATH_CALUDE_factorization_proof_l45_4558

theorem factorization_proof (b : ℝ) : 65 * b^2 + 195 * b = 65 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l45_4558


namespace NUMINAMATH_CALUDE_fourth_group_trees_l45_4583

theorem fourth_group_trees (total_groups : Nat) (average_trees : Nat)
  (group1_trees group2_trees group3_trees group5_trees : Nat) :
  total_groups = 5 →
  average_trees = 13 →
  group1_trees = 12 →
  group2_trees = 15 →
  group3_trees = 12 →
  group5_trees = 11 →
  (group1_trees + group2_trees + group3_trees + 15 + group5_trees) / total_groups = average_trees :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_group_trees_l45_4583


namespace NUMINAMATH_CALUDE_total_lunch_cost_l45_4516

/-- The total cost of lunches for a field trip --/
theorem total_lunch_cost : 
  let num_children : ℕ := 35
  let num_chaperones : ℕ := 5
  let num_teacher : ℕ := 1
  let num_additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_lunches : ℕ := num_children + num_chaperones + num_teacher + num_additional
  total_lunches * cost_per_lunch = 308 :=
by sorry

end NUMINAMATH_CALUDE_total_lunch_cost_l45_4516


namespace NUMINAMATH_CALUDE_kitchen_area_is_265_l45_4541

def total_area : ℕ := 1110
def num_bedrooms : ℕ := 4
def bedroom_length : ℕ := 11
def num_bathrooms : ℕ := 2
def bathroom_length : ℕ := 6
def bathroom_width : ℕ := 8

def bedroom_area : ℕ := bedroom_length * bedroom_length
def bathroom_area : ℕ := bathroom_length * bathroom_width
def total_bedroom_area : ℕ := num_bedrooms * bedroom_area
def total_bathroom_area : ℕ := num_bathrooms * bathroom_area
def remaining_area : ℕ := total_area - (total_bedroom_area + total_bathroom_area)

theorem kitchen_area_is_265 : remaining_area / 2 = 265 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_area_is_265_l45_4541


namespace NUMINAMATH_CALUDE_calculate_expression_l45_4502

theorem calculate_expression : 
  Real.sqrt 5 * (Real.sqrt 10 + 2) - 1 / (Real.sqrt 5 - 2) - Real.sqrt (1/2) = 
  (9 * Real.sqrt 2) / 2 + Real.sqrt 5 - 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l45_4502


namespace NUMINAMATH_CALUDE_g_difference_l45_4597

/-- The function g(n) as defined in the problem -/
def g (n : ℤ) : ℚ := (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

/-- Theorem stating the difference between g(m) and g(m-1) -/
theorem g_difference (m : ℤ) : g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l45_4597


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l45_4530

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

/-- The center of a circle -/
def Center : ℝ × ℝ := (-2, 0)

/-- The radius of a circle -/
def Radius : ℝ := 2

/-- Theorem: The circle described by x^2 + y^2 + 4x = 0 has center (-2, 0) and radius 2 -/
theorem circle_center_and_radius :
  (∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + y^2 = 4) ∧
  Center = (-2, 0) ∧
  Radius = 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l45_4530


namespace NUMINAMATH_CALUDE_cost_of_jeans_and_shirts_l45_4521

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := 9

theorem cost_of_jeans_and_shirts :
  3 * jeans_cost + 2 * shirt_cost = 69 :=
by
  have h1 : 2 * jeans_cost + 3 * shirt_cost = 61 := sorry
  sorry

end NUMINAMATH_CALUDE_cost_of_jeans_and_shirts_l45_4521


namespace NUMINAMATH_CALUDE_mitchs_family_milk_consumption_l45_4565

/-- The total milk consumption of Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk almond_milk oat_milk : ℝ) : ℝ :=
  regular_milk + soy_milk + almond_milk + oat_milk

/-- Theorem stating the total milk consumption of Mitch's family -/
theorem mitchs_family_milk_consumption :
  total_milk_consumption 1.75 0.85 1.25 0.65 = 4.50 := by
  sorry

end NUMINAMATH_CALUDE_mitchs_family_milk_consumption_l45_4565


namespace NUMINAMATH_CALUDE_littering_citations_l45_4589

/-- Represents the number of citations for each category --/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  smoking : ℕ
  parking : ℕ
  camping : ℕ

/-- Conditions for the park warden's citations --/
def citationConditions (c : Citations) : Prop :=
  c.littering = c.offLeash ∧
  c.littering = c.smoking + 5 ∧
  c.parking = 5 * (c.littering + c.offLeash + c.smoking) ∧
  c.camping = 10 ∧
  c.littering + c.offLeash + c.smoking + c.parking + c.camping = 150

/-- Theorem stating that under the given conditions, the number of littering citations is 9 --/
theorem littering_citations (c : Citations) (h : citationConditions c) : c.littering = 9 := by
  sorry


end NUMINAMATH_CALUDE_littering_citations_l45_4589


namespace NUMINAMATH_CALUDE_justin_tim_games_l45_4508

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- Justin and Tim are two specific players -/
def justin_and_tim : ℕ := 2

/-- The number of remaining players after Justin and Tim -/
def remaining_players : ℕ := total_players - justin_and_tim

/-- The number of additional players needed in a game with Justin and Tim -/
def additional_players : ℕ := players_per_game - justin_and_tim

theorem justin_tim_games (total_players : ℕ) (players_per_game : ℕ) (justin_and_tim : ℕ) 
  (remaining_players : ℕ) (additional_players : ℕ) :
  total_players = 12 →
  players_per_game = 6 →
  justin_and_tim = 2 →
  remaining_players = total_players - justin_and_tim →
  additional_players = players_per_game - justin_and_tim →
  Nat.choose remaining_players additional_players = 210 :=
by sorry

end NUMINAMATH_CALUDE_justin_tim_games_l45_4508


namespace NUMINAMATH_CALUDE_shifted_parabola_l45_4515

/-- The equation of a parabola shifted 1 unit to the left -/
theorem shifted_parabola (x y : ℝ) :
  (y = -(x^2) + 1) → 
  (∃ x' y', y' = -(x'^2) + 1 ∧ x' = x + 1) →
  y = -((x + 1)^2) + 1 := by
sorry

end NUMINAMATH_CALUDE_shifted_parabola_l45_4515


namespace NUMINAMATH_CALUDE_polygon_area_is_144_l45_4538

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Our specific polygon -/
def our_polygon : PerpendicularPolygon where
  sides := 36
  side_length := 2
  perimeter := 72
  area := 144

theorem polygon_area_is_144 (p : PerpendicularPolygon) 
  (h1 : p.sides = 36) 
  (h2 : p.perimeter = 72) 
  (h3 : p.side_length = p.perimeter / p.sides) : 
  p.area = 144 := by
  sorry

#check polygon_area_is_144

end NUMINAMATH_CALUDE_polygon_area_is_144_l45_4538


namespace NUMINAMATH_CALUDE_molecular_weight_CaOH2_is_74_10_l45_4524

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- The number of oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- The number of hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- The molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_CaOH2_is_74_10 :
  molecular_weight_CaOH2 = 74.10 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaOH2_is_74_10_l45_4524


namespace NUMINAMATH_CALUDE_rectangular_field_width_l45_4529

/-- Proves that for a rectangular field with length 7/5 of its width and perimeter 336 meters, the width is 70 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 336 →
  perimeter = 2 * length + 2 * width →
  width = 70 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l45_4529


namespace NUMINAMATH_CALUDE_chris_leftover_money_l45_4517

theorem chris_leftover_money (
  video_game_cost : ℕ)
  (candy_cost : ℕ)
  (babysitting_rate : ℕ)
  (hours_worked : ℕ)
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  babysitting_rate * hours_worked - (video_game_cost + candy_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_chris_leftover_money_l45_4517


namespace NUMINAMATH_CALUDE_distance_is_60_l45_4527

/-- The distance between a boy's house and school. -/
def distance : ℝ := sorry

/-- The time it takes for the boy to reach school when arriving on time. -/
def on_time : ℝ := sorry

/-- Assertion that when traveling at 10 km/hr, the boy arrives 2 hours late. -/
axiom late_arrival : on_time + 2 = distance / 10

/-- Assertion that when traveling at 20 km/hr, the boy arrives 1 hour early. -/
axiom early_arrival : on_time - 1 = distance / 20

/-- Theorem stating that the distance between the boy's house and school is 60 kilometers. -/
theorem distance_is_60 : distance = 60 := by sorry

end NUMINAMATH_CALUDE_distance_is_60_l45_4527


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l45_4591

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 60 →
  difference = 22 →
  us_count = total - difference →
  us_count = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l45_4591


namespace NUMINAMATH_CALUDE_amelia_painted_faces_l45_4506

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Amelia painted -/
def number_of_cuboids : ℕ := 6

/-- The total number of faces painted by Amelia -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem amelia_painted_faces :
  total_faces_painted = 36 :=
by sorry

end NUMINAMATH_CALUDE_amelia_painted_faces_l45_4506


namespace NUMINAMATH_CALUDE_smallest_valid_k_l45_4542

def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

def is_valid_k(k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ sum_to k = sum_to n - sum_to k

theorem smallest_valid_k :
  (∀ k : ℕ, k > 6 ∧ k < 9 → ¬is_valid_k k) ∧
  is_valid_k 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_k_l45_4542


namespace NUMINAMATH_CALUDE_remainder_problem_l45_4576

theorem remainder_problem : (7 * 10^20 + 2^20 + 5) % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l45_4576


namespace NUMINAMATH_CALUDE_correct_average_weight_l45_4557

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (actual_weight : ℝ) : 
  n = 20 → 
  initial_average = 58.4 → 
  misread_weight = 56 → 
  actual_weight = 68 → 
  (n * initial_average + actual_weight - misread_weight) / n = 59 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l45_4557


namespace NUMINAMATH_CALUDE_impossible_number_composition_l45_4501

def is_base_five_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 45

def compose_number (base_numbers : List ℕ) : ℕ := sorry

theorem impossible_number_composition :
  ¬ ∃ (x : ℕ) (base_numbers : List ℕ) (p q : ℕ),
    (base_numbers.length = 2021) ∧
    (∀ n ∈ base_numbers, is_base_five_two_digit n) ∧
    (∀ i, i < 2021 → i % 2 = 0 →
      base_numbers.get! i = base_numbers.get! (i + 1) - 1) ∧
    (x = compose_number base_numbers) ∧
    (Nat.Prime p ∧ Nat.Prime q) ∧
    (p * q = x) ∧
    (q = p + 2) :=
  sorry

end NUMINAMATH_CALUDE_impossible_number_composition_l45_4501


namespace NUMINAMATH_CALUDE_hall_length_l45_4536

/-- The length of a hall given its breadth, number of stones, and stone dimensions -/
theorem hall_length (breadth : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  breadth = 15 ∧ 
  num_stones = 5400 ∧
  stone_length = 0.2 ∧
  stone_width = 0.5 →
  (num_stones * stone_length * stone_width) / breadth = 36 := by
sorry


end NUMINAMATH_CALUDE_hall_length_l45_4536


namespace NUMINAMATH_CALUDE_snack_sales_averages_l45_4572

/-- Represents the snack sales data for a special weekend event -/
structure EventData where
  tickets : ℕ
  crackers : ℕ
  crackerPrice : ℚ
  beverages : ℕ
  beveragePrice : ℚ
  chocolates : ℕ
  chocolatePrice : ℚ

/-- Calculates the total snack sales for an event -/
def totalSales (e : EventData) : ℚ :=
  e.crackers * e.crackerPrice + e.beverages * e.beveragePrice + e.chocolates * e.chocolatePrice

/-- Calculates the average snack sales per ticket for an event -/
def averageSales (e : EventData) : ℚ :=
  totalSales e / e.tickets

/-- Theorem stating the average snack sales for each event and the combined average -/
theorem snack_sales_averages 
  (valentines : EventData)
  (stPatricks : EventData)
  (christmas : EventData)
  (h1 : valentines = ⟨10, 4, 11/5, 6, 3/2, 7, 6/5⟩)
  (h2 : stPatricks = ⟨8, 3, 2, 5, 25/20, 8, 1⟩)
  (h3 : christmas = ⟨9, 6, 43/20, 4, 17/12, 9, 11/10⟩) :
  averageSales valentines = 131/50 ∧
  averageSales stPatricks = 253/100 ∧
  averageSales christmas = 79/25 ∧
  (totalSales valentines + totalSales stPatricks + totalSales christmas) / 
  (valentines.tickets + stPatricks.tickets + christmas.tickets) = 139/50 := by
  sorry

end NUMINAMATH_CALUDE_snack_sales_averages_l45_4572


namespace NUMINAMATH_CALUDE_pop_expenditure_l45_4553

theorem pop_expenditure (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop →
  pop = 15 := by
sorry

end NUMINAMATH_CALUDE_pop_expenditure_l45_4553


namespace NUMINAMATH_CALUDE_sector_radius_l45_4526

/-- The radius of a circle given the area of a sector and its central angle -/
theorem sector_radius (area : ℝ) (angle : ℝ) (pi : ℝ) (h1 : area = 52.8) (h2 : angle = 42) 
  (h3 : pi = Real.pi) (h4 : area = (angle / 360) * pi * (radius ^ 2)) : radius = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l45_4526


namespace NUMINAMATH_CALUDE_largest_inscribed_square_l45_4568

theorem largest_inscribed_square (outer_square_side : ℝ) 
  (h_outer_square : outer_square_side = 12) : ℝ :=
  let triangle_side := 4 * Real.sqrt 6
  let inscribed_square_side := 6 - 2 * Real.sqrt 3
  inscribed_square_side

#check largest_inscribed_square

end NUMINAMATH_CALUDE_largest_inscribed_square_l45_4568


namespace NUMINAMATH_CALUDE_courtyard_length_l45_4577

/-- Prove that the length of a rectangular courtyard is 70 meters -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  width = 16.5 ∧ 
  num_stones = 231 ∧ 
  stone_length = 2.5 ∧ 
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 70 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l45_4577


namespace NUMINAMATH_CALUDE_initial_water_fraction_in_larger_jar_l45_4509

theorem initial_water_fraction_in_larger_jar 
  (small_capacity large_capacity : ℝ) 
  (h1 : small_capacity > 0) 
  (h2 : large_capacity > 0) 
  (h3 : small_capacity ≠ large_capacity) :
  let water_amount := (1/5) * small_capacity
  let initial_fraction := water_amount / large_capacity
  let combined_fraction := (water_amount + water_amount) / large_capacity
  (combined_fraction = 0.4) → (initial_fraction = 1/10) := by
  sorry

end NUMINAMATH_CALUDE_initial_water_fraction_in_larger_jar_l45_4509


namespace NUMINAMATH_CALUDE_only_C_is_random_event_l45_4580

-- Define the structure for an event
structure Event where
  description : String
  is_possible : Bool
  is_certain : Bool

-- Define the events
def event_A : Event := ⟨"Scoring 105 points in a percentile-based exam", false, false⟩
def event_B : Event := ⟨"Area of a rectangle with sides a and b is ab", true, true⟩
def event_C : Event := ⟨"Taking out 2 parts from 100 parts (2 defective, 98 non-defective), both are defective", true, false⟩
def event_D : Event := ⟨"Tossing a coin, it lands with either heads or tails up", true, true⟩

-- Define what a random event is
def is_random_event (e : Event) : Prop := e.is_possible ∧ ¬e.is_certain

-- Theorem stating that only event C is a random event
theorem only_C_is_random_event : 
  ¬is_random_event event_A ∧ 
  ¬is_random_event event_B ∧ 
  is_random_event event_C ∧ 
  ¬is_random_event event_D := by sorry

end NUMINAMATH_CALUDE_only_C_is_random_event_l45_4580


namespace NUMINAMATH_CALUDE_system_unique_solution_l45_4561

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y - 13)^2) + Real.sqrt ((x - 18)^2 + (y - 4)^2) = 15 ∧
  (x - 2*a)^2 + (y - 4*a)^2 = 1/4

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ :=
  {a | a = 145/44 ∨ a = 135/44 ∨ (63/20 < a ∧ a < 13/4)}

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∃! p : ℝ × ℝ, system p.1 p.2 a) ↔ a ∈ unique_solution_set :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l45_4561


namespace NUMINAMATH_CALUDE_tan_product_equals_fifteen_l45_4549

theorem tan_product_equals_fifteen : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_fifteen_l45_4549


namespace NUMINAMATH_CALUDE_intersection_A_B_l45_4556

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≥ 1}
def B : Set ℝ := {x : ℝ | 1 / x < 1}

-- State the theorem
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l45_4556


namespace NUMINAMATH_CALUDE_fractional_equation_root_l45_4537

theorem fractional_equation_root (x m : ℝ) : 
  ((x - 5) / (x + 2) = m / (x + 2) ∧ x + 2 ≠ 0) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l45_4537


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l45_4585

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 5

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem pirate_treasure_probability :
  (num_islands.choose num_treasure_islands) * 
  (prob_treasure ^ num_treasure_islands) * 
  (prob_neither ^ (num_islands - num_treasure_islands)) = 7/432 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l45_4585


namespace NUMINAMATH_CALUDE_row_1007_sum_equals_2013_squared_l45_4511

/-- The sum of numbers in the nth row of the given pattern -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the 1007th row sum equals 2013² -/
theorem row_1007_sum_equals_2013_squared :
  row_sum 1007 = 2013 ^ 2 := by sorry

end NUMINAMATH_CALUDE_row_1007_sum_equals_2013_squared_l45_4511


namespace NUMINAMATH_CALUDE_final_state_values_l45_4593

/-- Represents the state of variables a, b, and c -/
structure State :=
  (a : Int) (b : Int) (c : Int)

/-- Applies the sequence of operations to the initial state -/
def applyOperations (initial : State) : State :=
  let step1 := State.mk initial.b initial.b initial.c
  let step2 := State.mk step1.a step1.c step1.b
  State.mk step2.a step2.b step2.a

/-- The theorem stating the final values after operations -/
theorem final_state_values (initial : State := State.mk 3 (-5) 8) :
  let final := applyOperations initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_final_state_values_l45_4593


namespace NUMINAMATH_CALUDE_tetrahedral_pile_remaining_marbles_l45_4595

/-- The number of marbles in a tetrahedral pile of height k -/
def tetrahedralPile (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- The total number of marbles -/
def totalMarbles : ℕ := 60

/-- The height of the largest possible tetrahedral pile -/
def maxHeight : ℕ := 6

/-- The number of remaining marbles -/
def remainingMarbles : ℕ := totalMarbles - tetrahedralPile maxHeight

theorem tetrahedral_pile_remaining_marbles :
  remainingMarbles = 4 := by sorry

end NUMINAMATH_CALUDE_tetrahedral_pile_remaining_marbles_l45_4595


namespace NUMINAMATH_CALUDE_sequence_product_l45_4518

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_product (a b m n : ℝ) :
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l45_4518


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l45_4546

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1 ∧ x₂ = 3 ∧ 
  (x₁^2 - 4*x₁ + 3 = 0) ∧ 
  (x₂^2 - 4*x₂ + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l45_4546


namespace NUMINAMATH_CALUDE_largest_number_bound_l45_4578

theorem largest_number_bound (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 143)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 143 * 17 * 23 * 31 * k) :
  max a b ≤ 143 * 17 * 23 * 31 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_bound_l45_4578


namespace NUMINAMATH_CALUDE_park_boats_l45_4540

theorem park_boats (total_boats : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) 
  (h1 : total_boats = 42)
  (h2 : large_capacity = 6)
  (h3 : small_capacity = 4)
  (h4 : ∃ (large_boats small_boats : ℕ), 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats) :
  ∃ (large_boats small_boats : ℕ), 
    large_boats = 24 ∧ 
    small_boats = 18 ∧ 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats :=
by sorry

end NUMINAMATH_CALUDE_park_boats_l45_4540


namespace NUMINAMATH_CALUDE_sequence_sum_l45_4510

theorem sequence_sum : ∀ (a b c d : ℕ), 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = b * d) →  -- geometric progression
  (d = a + 50) →     -- difference between first and fourth terms
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) →  -- positive integers
  a + b + c + d = 215 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l45_4510


namespace NUMINAMATH_CALUDE_median_sufficiency_for_top_half_l45_4551

theorem median_sufficiency_for_top_half (scores : Finset ℝ) (xiaofen_score : ℝ) :
  Finset.card scores = 12 →
  Finset.card (Finset.filter (λ x => x = xiaofen_score) scores) ≤ 1 →
  (∃ median : ℝ, Finset.card (Finset.filter (λ x => x ≤ median) scores) = 6 ∧
                 Finset.card (Finset.filter (λ x => x ≥ median) scores) = 6) →
  (xiaofen_score > median ↔ Finset.card (Finset.filter (λ x => x > xiaofen_score) scores) < 6) :=
by sorry

end NUMINAMATH_CALUDE_median_sufficiency_for_top_half_l45_4551


namespace NUMINAMATH_CALUDE_count_nines_to_hundred_l45_4504

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for numbers from 1 to n -/
def sum_nines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the count of 9s in numbers from 1 to 100 is 19 -/
theorem count_nines_to_hundred : sum_nines 100 = 19 := by sorry

end NUMINAMATH_CALUDE_count_nines_to_hundred_l45_4504


namespace NUMINAMATH_CALUDE_binary_11111011111_equals_2015_l45_4566

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_11111011111_equals_2015 :
  binary_to_decimal [true, true, true, true, true, false, true, true, true, true, true] = 2015 := by
  sorry

end NUMINAMATH_CALUDE_binary_11111011111_equals_2015_l45_4566


namespace NUMINAMATH_CALUDE_anns_number_l45_4552

theorem anns_number (y : ℚ) : 5 * (3 * y + 15) = 200 → y = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_anns_number_l45_4552


namespace NUMINAMATH_CALUDE_smallest_d_for_10000_l45_4522

theorem smallest_d_for_10000 : 
  ∃ (p q r : Nat), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (∀ d : Nat, d > 0 → 
      (∃ (p' q' r' : Nat), 
        Prime p' ∧ Prime q' ∧ Prime r' ∧ 
        p' ≠ q' ∧ p' ≠ r' ∧ q' ≠ r' ∧
        10000 * d = (p' * q' * r')^2) → 
      d ≥ 53361) ∧
    10000 * 53361 = (p * q * r)^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_10000_l45_4522


namespace NUMINAMATH_CALUDE_monkey_climbing_time_l45_4532

/-- Monkey climbing problem -/
theorem monkey_climbing_time (tree_height : ℕ) (climb_rate : ℕ) (slip_rate : ℕ) : 
  tree_height = 22 ∧ climb_rate = 3 ∧ slip_rate = 2 → 
  (tree_height - 1) / (climb_rate - slip_rate) + 1 = 22 := by
sorry

end NUMINAMATH_CALUDE_monkey_climbing_time_l45_4532


namespace NUMINAMATH_CALUDE_f_minus_five_l45_4555

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_five (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_f_minus_one : f (-1) = 1) :
  f (-5) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_five_l45_4555


namespace NUMINAMATH_CALUDE_max_value_of_a_l45_4545

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l45_4545


namespace NUMINAMATH_CALUDE_range_of_circle_l45_4570

theorem range_of_circle (x y : ℝ) (h : x^2 + y^2 = 4*x) :
  ∃ (z : ℝ), z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_range_of_circle_l45_4570


namespace NUMINAMATH_CALUDE_exists_min_value_in_interval_l45_4543

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem exists_min_value_in_interval :
  ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 1 → m ≤ f x :=
sorry

end NUMINAMATH_CALUDE_exists_min_value_in_interval_l45_4543


namespace NUMINAMATH_CALUDE_distinct_cube_configurations_l45_4533

-- Define the cube configuration
structure CubeConfig where
  white : Fin 8 → Fin 3
  blue : Fin 8 → Fin 3
  red : Fin 8 → Fin 2

-- Define the rotation group
def RotationGroup : Type := Unit

-- Define the action of the rotation group on cube configurations
def rotate : RotationGroup → CubeConfig → CubeConfig := sorry

-- Define the orbit of a cube configuration under rotations
def orbit (c : CubeConfig) : Set CubeConfig := sorry

-- Define the set of all valid cube configurations
def AllConfigs : Set CubeConfig := sorry

-- Count the number of distinct orbits
def countDistinctOrbits : ℕ := sorry

-- The main theorem
theorem distinct_cube_configurations :
  countDistinctOrbits = 25 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_configurations_l45_4533


namespace NUMINAMATH_CALUDE_solutions_of_equation_l45_4528

theorem solutions_of_equation (x : ℝ) : 
  (3 * x^2 = Real.sqrt 3 * x) ↔ (x = 0 ∨ x = Real.sqrt 3 / 3) := by
sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l45_4528


namespace NUMINAMATH_CALUDE_paper_towel_cost_l45_4574

theorem paper_towel_cost (case_price : ℝ) (savings_percent : ℝ) (rolls_per_case : ℕ) : 
  case_price = 9 ∧ savings_percent = 25 ∧ rolls_per_case = 12 →
  (case_price / (1 - savings_percent / 100)) / rolls_per_case = 0.9375 := by
sorry

end NUMINAMATH_CALUDE_paper_towel_cost_l45_4574


namespace NUMINAMATH_CALUDE_exists_polynomial_for_cosine_multiple_l45_4588

-- Define Chebyshev polynomials of the first kind
def chebyshev (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | 1 => λ x => x
  | n + 2 => λ x => 2 * x * chebyshev (n + 1) x - chebyshev n x

-- State the theorem
theorem exists_polynomial_for_cosine_multiple (n : ℕ) (hn : n > 0) :
  ∃ (p : ℝ → ℝ), ∀ x, p (2 * Real.cos x) = 2 * Real.cos (n * x) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_for_cosine_multiple_l45_4588


namespace NUMINAMATH_CALUDE_michael_record_score_l45_4567

/-- Given a basketball team's total score and the average score of other players,
    calculate Michael's score that set the new school record. -/
theorem michael_record_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) 
    (h1 : total_score = 75)
    (h2 : other_players = 5)
    (h3 : avg_score = 6) :
    total_score - (other_players * avg_score) = 45 := by
  sorry

#check michael_record_score

end NUMINAMATH_CALUDE_michael_record_score_l45_4567


namespace NUMINAMATH_CALUDE_power_multiplication_l45_4573

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l45_4573


namespace NUMINAMATH_CALUDE_scout_troop_profit_l45_4525

/-- The profit of a scout troop selling candy bars -/
theorem scout_troop_profit :
  let num_bars : ℕ := 1200
  let buy_price : ℚ := 1 / 3  -- price per bar when buying
  let sell_price : ℚ := 3 / 5 -- price per bar when selling
  let cost : ℚ := num_bars * buy_price
  let revenue : ℚ := num_bars * sell_price
  let profit : ℚ := revenue - cost
  profit = 320 := by sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l45_4525


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l45_4587

/-- Given a cosine function with a phase shift, prove that under certain symmetry conditions,
    the symmetric center closest to the origin is at a specific point when the period is maximized. -/
theorem cosine_symmetry_center (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.cos (ω * x + 3 * Real.pi / 4)
  (∀ x : ℝ, f (π / 3 - x) = f (π / 3 + x)) →  -- Symmetry about x = π/6
  (∀ k : ℤ, ω ≠ 6 * k - 9 / 2 → ω > 6 * k - 9 / 2) →  -- ω is the smallest positive value
  (π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } →  -- Symmetric center formula
  (-π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } ∧  -- Closest symmetric center
  (∀ x : ℝ, x ∈ { y : ℝ | ∃ k : ℤ, y = 2 / 3 * k * π - π / 6 } → |x| ≥ |(-π / 6 : ℝ)|) :=
by
  sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l45_4587


namespace NUMINAMATH_CALUDE_bird_lake_swans_l45_4560

theorem bird_lake_swans (total_birds : ℕ) (duck_fraction : ℚ) : 
  total_birds = 108 →
  duck_fraction = 5/6 →
  (1 - duck_fraction) * total_birds = 18 :=
by sorry

end NUMINAMATH_CALUDE_bird_lake_swans_l45_4560


namespace NUMINAMATH_CALUDE_parabola_properties_l45_4503

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (-1, -8)

-- Define the shift
def m : ℝ := 3

-- Theorem statement
theorem parabola_properties :
  (∀ x, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 ∧
  parabola (m - 0) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l45_4503


namespace NUMINAMATH_CALUDE_same_color_shoe_probability_l45_4547

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The probability of selecting two shoes of the same color -/
def prob_same_color : ℚ := 9 / 2601

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes - 1) / (total_shoes.choose 2) = prob_same_color := by
  sorry

end NUMINAMATH_CALUDE_same_color_shoe_probability_l45_4547


namespace NUMINAMATH_CALUDE_range_of_f_range_of_g_l45_4554

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem for part (1)
theorem range_of_f (a : ℝ) :
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ a = -1 ∨ a = 3/2 :=
sorry

-- Theorem for part (2)
theorem range_of_g :
  (∀ a x : ℝ, f a x ≥ 0) →
  ∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 ↔ ∃ a : ℝ, g a = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_g_l45_4554


namespace NUMINAMATH_CALUDE_certain_number_is_six_l45_4519

theorem certain_number_is_six (a b n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a % n = 2) (h5 : b % n = 3) (h6 : (a - b) % n = 5) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l45_4519


namespace NUMINAMATH_CALUDE_stream_speed_l45_4531

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream --/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 10)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l45_4531


namespace NUMINAMATH_CALUDE_abc_inequality_l45_4599

theorem abc_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l45_4599


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l45_4520

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l45_4520


namespace NUMINAMATH_CALUDE_discount_ticket_price_l45_4596

theorem discount_ticket_price (discount_rate : ℝ) (discounted_price : ℝ) (original_price : ℝ) :
  discount_rate = 0.3 →
  discounted_price = 1400 →
  discounted_price = (1 - discount_rate) * original_price →
  original_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_discount_ticket_price_l45_4596


namespace NUMINAMATH_CALUDE_sum_in_base6_l45_4562

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given numbers in base 6 equals 1214 in base 6 -/
theorem sum_in_base6 :
  let n1 := [5, 5, 5]
  let n2 := [5, 5]
  let n3 := [5]
  let n4 := [1, 1, 1]
  let sum := base6To10 n1 + base6To10 n2 + base6To10 n3 + base6To10 n4
  base10To6 sum = [1, 2, 1, 4] :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l45_4562


namespace NUMINAMATH_CALUDE_power_division_simplification_l45_4523

theorem power_division_simplification (a : ℝ) : (2 * a) ^ 7 / (2 * a) ^ 4 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l45_4523


namespace NUMINAMATH_CALUDE_cube_difference_l45_4594

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 208 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l45_4594


namespace NUMINAMATH_CALUDE_age_problem_l45_4569

/-- Given three people whose total present age is 90 years and whose ages were in the ratio 1:2:3 ten years ago, 
    the present age of the person who was in the middle of the ratio is 30 years. -/
theorem age_problem (a b c : ℕ) : 
  a + b + c = 90 →  -- Total present age is 90
  (a - 10) = (b - 10) / 2 →  -- Ratio condition for a and b
  (c - 10) = 3 * ((b - 10) / 2) →  -- Ratio condition for b and c
  b = 30 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l45_4569


namespace NUMINAMATH_CALUDE_curves_with_property_P_l45_4550

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

-- Define property P
def property_P (curve : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∃ A B : ℝ × ℝ, 
    curve A.1 A.2 ∧ curve B.1 B.2 ∧
    line_equation k A.1 A.2 ∧ line_equation k B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = k^2

-- Define the three curves
def curve1 (x y : ℝ) : Prop := y = -abs x

def curve2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

def curve3 (x y : ℝ) : Prop := y = (x + 1)^2

-- Theorem statement
theorem curves_with_property_P :
  ¬(property_P curve1) ∧ 
  property_P curve2 ∧ 
  property_P curve3 :=
sorry

end NUMINAMATH_CALUDE_curves_with_property_P_l45_4550


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l45_4590

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l45_4590


namespace NUMINAMATH_CALUDE_percentage_of_difference_l45_4579

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P / 100 * (x - y) = 15 / 100 * (x + y) →
  y = 14.285714285714285 / 100 * x →
  P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l45_4579


namespace NUMINAMATH_CALUDE_choose_from_two_bags_l45_4534

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (m n : ℕ) : ℕ := m * n

/-- The number of balls in the red bag -/
def red_balls : ℕ := 3

/-- The number of balls in the blue bag -/
def blue_balls : ℕ := 5

/-- Theorem: The number of ways to choose one ball from the red bag and one from the blue bag is 15 -/
theorem choose_from_two_bags : choose_one_from_each red_balls blue_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_from_two_bags_l45_4534


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l45_4544

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  b = 80 :=  -- The smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l45_4544


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l45_4548

/-- Given an arithmetic sequence {a_n} where a_4 = 2, the maximum value of a_2 * a_6 is 4. -/
theorem arithmetic_sequence_max_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 4 = 2 →                                        -- given condition
  ∃ (x : ℝ), x = a 2 * a 6 ∧ x ≤ 4 ∧ 
  ∀ (y : ℝ), y = a 2 * a 6 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l45_4548
