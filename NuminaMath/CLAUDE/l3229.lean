import Mathlib

namespace minimum_width_proof_l3229_322961

/-- Represents the width of the rectangular fence -/
def width : ℝ → ℝ := λ w => w

/-- Represents the length of the rectangular fence -/
def length : ℝ → ℝ := λ w => w + 20

/-- Represents the area of the rectangular fence -/
def area : ℝ → ℝ := λ w => width w * length w

/-- Represents the perimeter of the rectangular fence -/
def perimeter : ℝ → ℝ := λ w => 2 * (width w + length w)

/-- The minimum width of the rectangular fence that satisfies the given conditions -/
def min_width : ℝ := 10

theorem minimum_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → perimeter min_width ≤ perimeter w) ∧
  area min_width ≥ 200 := by sorry

end minimum_width_proof_l3229_322961


namespace root_implies_q_value_l3229_322930

theorem root_implies_q_value (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (3 + 2 * Complex.I)^2 + p * (3 + 2 * Complex.I) + q = 0 →
  q = 26 := by
sorry

end root_implies_q_value_l3229_322930


namespace sum_of_powers_of_3_mod_5_l3229_322926

def sum_of_powers (base : ℕ) (exponent : ℕ) : ℕ :=
  Finset.sum (Finset.range (exponent + 1)) (fun i => base ^ i)

theorem sum_of_powers_of_3_mod_5 :
  sum_of_powers 3 2023 % 5 = 3 := by sorry

end sum_of_powers_of_3_mod_5_l3229_322926


namespace training_cost_calculation_l3229_322948

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℝ
  revenue : ℝ
  trainingMonths : ℕ
  hiringBonus : ℝ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) (trainingCostPerMonth : ℝ) : ℝ :=
  a.revenue - (a.salary + a.hiringBonus + a.trainingMonths * trainingCostPerMonth)

theorem training_cost_calculation (applicant1 applicant2 : Applicant) 
  (h1 : applicant1.salary = 42000)
  (h2 : applicant1.revenue = 93000)
  (h3 : applicant1.trainingMonths = 3)
  (h4 : applicant1.hiringBonus = 0)
  (h5 : applicant2.salary = 45000)
  (h6 : applicant2.revenue = 92000)
  (h7 : applicant2.trainingMonths = 0)
  (h8 : applicant2.hiringBonus = 0.01 * applicant2.salary)
  (h9 : ∃ (trainingCostPerMonth : ℝ), 
    netGain applicant1 trainingCostPerMonth - netGain applicant2 0 = 850 ∨
    netGain applicant2 0 - netGain applicant1 trainingCostPerMonth = 850) :
  ∃ (trainingCostPerMonth : ℝ), trainingCostPerMonth = 17866.67 := by
  sorry

end training_cost_calculation_l3229_322948


namespace max_playground_area_l3229_322989

/-- Represents the dimensions of a rectangular playground. -/
structure Playground where
  width : ℝ
  length : ℝ

/-- The total fencing available for the playground. -/
def totalFencing : ℝ := 480

/-- Calculates the area of a playground. -/
def area (p : Playground) : ℝ := p.width * p.length

/-- Checks if a playground satisfies the fencing constraint. -/
def satisfiesFencingConstraint (p : Playground) : Prop :=
  p.length + 2 * p.width = totalFencing

/-- Theorem stating the maximum area of the playground. -/
theorem max_playground_area :
  ∃ (p : Playground), satisfiesFencingConstraint p ∧
    area p = 28800 ∧
    ∀ (q : Playground), satisfiesFencingConstraint q → area q ≤ area p :=
sorry

end max_playground_area_l3229_322989


namespace ba_atomic_weight_l3229_322933

/-- The atomic weight of Bromine (Br) -/
def atomic_weight_Br : ℝ := 79.9

/-- The molecular weight of the compound BaBr₂ -/
def molecular_weight_compound : ℝ := 297

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_compound - 2 * atomic_weight_Br

theorem ba_atomic_weight :
  atomic_weight_Ba = 137.2 := by sorry

end ba_atomic_weight_l3229_322933


namespace no_solution_exists_l3229_322980

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  (a * b + 90 = 24 * Nat.lcm a b + 15 * Nat.gcd a b) ∧ 
  (Nat.gcd a b = 3) := by
  sorry

end no_solution_exists_l3229_322980


namespace santinos_garden_fruit_count_l3229_322923

/-- Represents the number of trees for each fruit type in Santino's garden -/
structure TreeCounts where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Represents the fruit production rate for each tree type -/
structure FruitProduction where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Calculates the total number of fruits in Santino's garden -/
def totalFruits (trees : TreeCounts) (production : FruitProduction) : ℕ :=
  trees.papaya * production.papaya +
  trees.mango * production.mango +
  trees.apple * production.apple +
  trees.orange * production.orange

theorem santinos_garden_fruit_count :
  let trees : TreeCounts := ⟨2, 3, 4, 5⟩
  let production : FruitProduction := ⟨10, 20, 15, 25⟩
  totalFruits trees production = 265 := by
  sorry

end santinos_garden_fruit_count_l3229_322923


namespace exactly_one_positive_integer_solution_l3229_322913

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 16 - 4 * n > 10 :=
by sorry

end exactly_one_positive_integer_solution_l3229_322913


namespace orthogonal_vectors_sum_l3229_322994

theorem orthogonal_vectors_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 :=
by sorry

end orthogonal_vectors_sum_l3229_322994


namespace xy_value_l3229_322919

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end xy_value_l3229_322919


namespace binary_101101_equals_base5_140_l3229_322958

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
by sorry

end binary_101101_equals_base5_140_l3229_322958


namespace sqrt_meaningful_range_l3229_322935

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end sqrt_meaningful_range_l3229_322935


namespace sum_of_square_areas_l3229_322934

theorem sum_of_square_areas (square1_side : ℝ) (square2_side : ℝ) 
  (h1 : square1_side = 11) (h2 : square2_side = 5) : 
  square1_side ^ 2 + square2_side ^ 2 = 146 := by
  sorry

end sum_of_square_areas_l3229_322934


namespace roots_expression_value_l3229_322978

theorem roots_expression_value (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 7*γ^4 + 10*δ^3 = 1363 := by
  sorry

end roots_expression_value_l3229_322978


namespace x_power_five_minus_reciprocal_l3229_322955

theorem x_power_five_minus_reciprocal (x : ℝ) (h : x + 1/x = Real.sqrt 2) :
  x^5 - 1/x^5 = 0 := by
  sorry

end x_power_five_minus_reciprocal_l3229_322955


namespace candy_bar_sales_difference_l3229_322975

/-- Candy bar sales problem -/
theorem candy_bar_sales_difference (price_a price_b : ℕ)
  (marvin_a marvin_b : ℕ) (tina_a tina_b : ℕ)
  (marvin_discount_threshold marvin_discount_amount : ℕ)
  (tina_discount_threshold tina_discount_amount : ℕ)
  (tina_returns : ℕ) :
  price_a = 2 →
  price_b = 3 →
  marvin_a = 20 →
  marvin_b = 15 →
  tina_a = 70 →
  tina_b = 35 →
  marvin_discount_threshold = 5 →
  marvin_discount_amount = 1 →
  tina_discount_threshold = 10 →
  tina_discount_amount = 2 →
  tina_returns = 2 →
  (tina_a * price_a + tina_b * price_b
    - (tina_b / tina_discount_threshold) * tina_discount_amount
    - tina_returns * price_b)
  - (marvin_a * price_a + marvin_b * price_b
    - (marvin_a / marvin_discount_threshold) * marvin_discount_amount)
  = 152 := by sorry

end candy_bar_sales_difference_l3229_322975


namespace apple_basket_count_l3229_322996

theorem apple_basket_count (rotten_percent : ℝ) (spotted_percent : ℝ) (insect_percent : ℝ) (varying_rot_percent : ℝ) (perfect_count : ℕ) : 
  rotten_percent = 0.12 →
  spotted_percent = 0.07 →
  insect_percent = 0.05 →
  varying_rot_percent = 0.03 →
  perfect_count = 66 →
  ∃ (total : ℕ), total = 90 ∧ 
    (1 - (rotten_percent + spotted_percent + insect_percent + varying_rot_percent)) * (total : ℝ) = perfect_count :=
by
  sorry

end apple_basket_count_l3229_322996


namespace plain_cookies_sold_l3229_322957

-- Define the types for our variables
def chocolate_chip_price : ℚ := 125 / 100
def plain_price : ℚ := 75 / 100
def total_boxes : ℕ := 1585
def total_value : ℚ := 158625 / 100

-- Define the theorem
theorem plain_cookies_sold :
  ∃ (c p : ℕ),
    c + p = total_boxes ∧
    c * chocolate_chip_price + p * plain_price = total_value ∧
    p = 790 := by
  sorry


end plain_cookies_sold_l3229_322957


namespace job_completion_time_l3229_322911

theorem job_completion_time (days : ℝ) (fraction_completed : ℝ) (h1 : fraction_completed = 5 / 8) (h2 : days = 10) :
  (days / fraction_completed) = 16 := by
  sorry

end job_completion_time_l3229_322911


namespace absolute_value_inequality_solution_set_l3229_322964

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 3| ≥ 4} = {x : ℝ | x ≤ -5/2} := by
  sorry

end absolute_value_inequality_solution_set_l3229_322964


namespace potato_planting_l3229_322909

theorem potato_planting (rows : ℕ) (additional_plants : ℕ) (total_plants : ℕ) 
  (h1 : rows = 7)
  (h2 : additional_plants = 15)
  (h3 : total_plants = 141)
  : (total_plants - additional_plants) / rows = 18 := by
  sorry

end potato_planting_l3229_322909


namespace existence_of_a_l3229_322916

theorem existence_of_a : ∃ a : ℝ, a ≥ 1 ∧ 
  (∀ x : ℝ, |x - 1| > a → Real.log (x^2 - 3*x + 3) > 0) ∧
  (∃ x : ℝ, Real.log (x^2 - 3*x + 3) > 0 ∧ |x - 1| ≤ a) :=
by sorry

end existence_of_a_l3229_322916


namespace game_cost_l3229_322937

/-- Given Frank's lawn mowing earnings, expenses, and game purchasing ability, prove the cost of each game. -/
theorem game_cost (total_earned : ℕ) (spent : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 19)
  (h2 : spent = 11)
  (h3 : num_games = 4)
  (h4 : ∃ (cost : ℕ), (total_earned - spent) = num_games * cost) :
  ∃ (cost : ℕ), cost = 2 ∧ (total_earned - spent) = num_games * cost := by
  sorry

end game_cost_l3229_322937


namespace cab_speed_reduction_l3229_322954

theorem cab_speed_reduction (usual_time : ℝ) (delay : ℝ) :
  usual_time = 75 ∧ delay = 15 →
  (usual_time / (usual_time + delay)) = 5 / 6 := by
sorry

end cab_speed_reduction_l3229_322954


namespace walnut_trees_planted_l3229_322910

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial final planted : ℕ) :
  initial = 22 →
  final = 55 →
  planted = final - initial →
  planted = 33 := by sorry

end walnut_trees_planted_l3229_322910


namespace percent_difference_l3229_322968

theorem percent_difference (y q w z : ℝ) 
  (hw : w = 0.6 * q)
  (hq : q = 0.6 * y)
  (hz : z = 0.54 * y) :
  z = w * 1.5 := by
sorry

end percent_difference_l3229_322968


namespace simplify_expression_l3229_322999

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - (1 / (1 + (a + 1) / (1 - a))) = (1 + a) / 2 := by sorry

end simplify_expression_l3229_322999


namespace number_relation_l3229_322981

theorem number_relation (A B : ℝ) (h : A = B * (1 + 0.1)) : B = A * (10/11) := by
  sorry

end number_relation_l3229_322981


namespace medicine_price_reduction_l3229_322993

theorem medicine_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 32)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price :=
by
  sorry

end medicine_price_reduction_l3229_322993


namespace ceiling_floor_difference_l3229_322924

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-35 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-35 : ℚ) / 4⌋⌋ = 1 := by
  sorry

end ceiling_floor_difference_l3229_322924


namespace rachels_age_l3229_322960

theorem rachels_age (rachel leah sam alex : ℝ) 
  (h1 : rachel = leah + 4)
  (h2 : rachel + leah = 2 * sam)
  (h3 : alex = 2 * rachel)
  (h4 : rachel + leah + sam + alex = 92) :
  rachel = 24.5 := by
  sorry

end rachels_age_l3229_322960


namespace polynomial_factorization_l3229_322949

theorem polynomial_factorization (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by sorry

end polynomial_factorization_l3229_322949


namespace wheel_radius_l3229_322956

theorem wheel_radius (total_distance : ℝ) (revolutions : ℕ) (h1 : total_distance = 798.2857142857142) (h2 : revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.254092376554174) < 0.000000000000001 :=
by
  sorry

end wheel_radius_l3229_322956


namespace lowest_price_scheme_l3229_322990

-- Define the pricing schemes
def schemeA (price : ℝ) : ℝ := price * 1.1 * 0.9
def schemeB (price : ℝ) : ℝ := price * 0.9 * 1.1
def schemeC (price : ℝ) : ℝ := price * 1.15 * 0.85
def schemeD (price : ℝ) : ℝ := price * 1.2 * 0.8

-- Theorem statement
theorem lowest_price_scheme (price : ℝ) (h : price > 0) :
  schemeD price = min (schemeA price) (min (schemeB price) (schemeC price)) :=
by sorry

end lowest_price_scheme_l3229_322990


namespace constant_term_expansion_l3229_322959

/-- The constant term in the expansion of (2x + 1/x)^6 -/
def constantTerm : ℕ := 160

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the expansion -/
def generalTerm (r : ℕ) : ℚ :=
  2^(6 - r) * binomial 6 r * (1 : ℚ)

theorem constant_term_expansion :
  constantTerm = generalTerm 3 := by sorry

end constant_term_expansion_l3229_322959


namespace spaceship_break_time_l3229_322922

/-- Represents the travel pattern of a spaceship -/
structure TravelPattern where
  initialTravel : ℕ
  initialBreak : ℕ
  secondTravel : ℕ
  secondBreak : ℕ
  subsequentTravel : ℕ
  subsequentBreak : ℕ

/-- Calculates the total break time for a spaceship journey -/
def calculateBreakTime (pattern : TravelPattern) (totalJourneyTime : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for the given travel pattern and journey time, 
    the total break time is 8 hours -/
theorem spaceship_break_time :
  let pattern : TravelPattern := {
    initialTravel := 10,
    initialBreak := 3,
    secondTravel := 10,
    secondBreak := 1,
    subsequentTravel := 11,
    subsequentBreak := 1
  }
  let totalJourneyTime : ℕ := 72
  calculateBreakTime pattern totalJourneyTime = 8 := by
  sorry

end spaceship_break_time_l3229_322922


namespace arithmetic_sequence_common_difference_l3229_322906

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r ≠ 0, ∀ n, a (n + 1) = r * a n

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geometric : geometric_sequence (λ n ↦ a (2 * n - 1)))
  (h_arithmetic : arithmetic_sequence a d) :
  d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l3229_322906


namespace symmetry_of_expressions_l3229_322921

-- Define a completely symmetric expression
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetry_of_expressions :
  is_completely_symmetric expr1 ∧ 
  is_completely_symmetric expr2 ∧ 
  ¬is_completely_symmetric expr3 := by
  sorry

end symmetry_of_expressions_l3229_322921


namespace badminton_players_l3229_322982

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 30)
  (h_tennis : tennis = 21)
  (h_neither : neither = 2)
  (h_both : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end badminton_players_l3229_322982


namespace infinitely_many_solutions_l3229_322915

theorem infinitely_many_solutions (c : ℚ) :
  (∀ y : ℚ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 := by
  sorry

end infinitely_many_solutions_l3229_322915


namespace nested_fraction_evaluation_l3229_322928

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (4 + 1 / 5)) = 21 / 26 := by
  sorry

end nested_fraction_evaluation_l3229_322928


namespace distribute_5_3_l3229_322905

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end distribute_5_3_l3229_322905


namespace smallest_divisible_by_15_18_20_l3229_322914

theorem smallest_divisible_by_15_18_20 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by
  use 180
  sorry

end smallest_divisible_by_15_18_20_l3229_322914


namespace jacket_final_price_l3229_322927

/-- The final price of a jacket after multiple discounts -/
theorem jacket_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) 
  (h1 : original_price = 25)
  (h2 : discount1 = 0.40)
  (h3 : discount2 = 0.25)
  (h4 : discount3 = 0.10) : 
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 10.125 := by
  sorry

end jacket_final_price_l3229_322927


namespace solve_for_b_l3229_322946

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 := by
  sorry

end solve_for_b_l3229_322946


namespace bakers_sales_comparison_l3229_322907

/-- Baker's sales comparison -/
theorem bakers_sales_comparison 
  (usual_pastries : ℕ) (usual_bread : ℕ) 
  (today_pastries : ℕ) (today_bread : ℕ) 
  (pastry_price : ℕ) (bread_price : ℕ) : 
  usual_pastries = 20 → 
  usual_bread = 10 → 
  today_pastries = 14 → 
  today_bread = 25 → 
  pastry_price = 2 → 
  bread_price = 4 → 
  (today_pastries * pastry_price + today_bread * bread_price) - 
  (usual_pastries * pastry_price + usual_bread * bread_price) = 48 := by
  sorry

end bakers_sales_comparison_l3229_322907


namespace quadratic_form_inequality_l3229_322966

theorem quadratic_form_inequality (a b c d : ℝ) (h : a * d - b * c = 1) :
  a^2 + b^2 + c^2 + d^2 + a * c + b * d > 1 := by
  sorry

end quadratic_form_inequality_l3229_322966


namespace monochromatic_rectangle_exists_l3229_322991

-- Define the colors
inductive Color
  | Red
  | Blue
  | Green

-- Define the board as a function from coordinates to colors
def Board := ℤ × ℤ → Color

-- Define a rectangle on the board
def IsRectangle (board : Board) (x1 y1 x2 y2 : ℤ) : Prop :=
  x1 ≠ x2 ∧ y1 ≠ y2 ∧
  board (x1, y1) = board (x2, y1) ∧
  board (x1, y1) = board (x1, y2) ∧
  board (x1, y1) = board (x2, y2)

-- The main theorem
theorem monochromatic_rectangle_exists (board : Board) :
  ∃ x1 y1 x2 y2 : ℤ, IsRectangle board x1 y1 x2 y2 := by
  sorry


end monochromatic_rectangle_exists_l3229_322991


namespace master_bathroom_towel_price_l3229_322904

/-- The price of towel sets for the master bathroom, given the following conditions:
  * 2 sets of towels for guest bathroom and 4 sets for master bathroom are bought
  * Guest bathroom towel sets cost $40.00 each
  * The store offers a 20% discount
  * The total spent on towel sets is $224
-/
theorem master_bathroom_towel_price :
  ∀ (x : ℝ),
    2 * 40 * (1 - 0.2) + 4 * x * (1 - 0.2) = 224 →
    x = 50 := by
  sorry

end master_bathroom_towel_price_l3229_322904


namespace order_of_cube_roots_l3229_322945

theorem order_of_cube_roots : ∀ (a b c : ℝ),
  a = 2^(4/3) →
  b = 3^(2/3) →
  c = 2.5^(1/3) →
  c < b ∧ b < a :=
by sorry

end order_of_cube_roots_l3229_322945


namespace inequality_system_solution_set_l3229_322931

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 - x > 0 ∧ 2*x + 3 > 1) ↔ (-1 < x ∧ x < 2) := by
  sorry

end inequality_system_solution_set_l3229_322931


namespace custom_mult_example_l3229_322903

-- Define the custom operation *
def custom_mult (a b : Int) : Int := a * b

-- Theorem statement
theorem custom_mult_example : custom_mult 2 (-3) = -6 := by
  sorry

end custom_mult_example_l3229_322903


namespace largest_divisor_of_consecutive_odd_product_l3229_322950

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), Even m → m > 0 →
    k ∣ (m + 1) * (m + 3) * (m + 5) * (m + 7) * (m + 9)) :=
by sorry

end largest_divisor_of_consecutive_odd_product_l3229_322950


namespace sufficient_necessary_condition_l3229_322969

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end sufficient_necessary_condition_l3229_322969


namespace hexagon_perimeter_l3229_322941

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by sorry

end hexagon_perimeter_l3229_322941


namespace max_remainder_when_divided_by_25_l3229_322944

theorem max_remainder_when_divided_by_25 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 25 * B + C →
  C ≤ 24 :=
by sorry

end max_remainder_when_divided_by_25_l3229_322944


namespace notebooks_given_to_yujeong_l3229_322920

/-- The number of notebooks Minyoung initially had -/
def initial_notebooks : ℕ := 17

/-- The number of notebooks Minyoung had left after giving some to Yujeong -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Minyoung gave to Yujeong -/
def notebooks_given : ℕ := initial_notebooks - remaining_notebooks

theorem notebooks_given_to_yujeong :
  notebooks_given = 9 :=
sorry

end notebooks_given_to_yujeong_l3229_322920


namespace expression_equality_l3229_322963

theorem expression_equality : 
  Real.sqrt 8 + |1 - Real.sqrt 2| - (1 / 2)⁻¹ + (π - Real.sqrt 3)^0 = 3 * Real.sqrt 2 - 2 := by
  sorry

end expression_equality_l3229_322963


namespace largest_triangle_perimeter_l3229_322992

theorem largest_triangle_perimeter :
  ∀ y : ℕ,
  y > 0 →
  y < 16 →
  7 + y > 9 →
  9 + y > 7 →
  (∀ z : ℕ, z > 0 → z < 16 → 7 + z > 9 → 9 + z > 7 → 7 + 9 + y ≥ 7 + 9 + z) →
  7 + 9 + y = 31 :=
by
  sorry

end largest_triangle_perimeter_l3229_322992


namespace reusable_bags_estimate_conditional_probability_second_spender_l3229_322962

/-- Represents the survey data for each age group -/
structure AgeGroupData :=
  (spent_more : Nat)  -- Number of people who spent ≥ $188
  (spent_less : Nat)  -- Number of people who spent < $188

/-- Represents the survey results -/
def survey_data : List AgeGroupData := [
  ⟨8, 2⟩,   -- [20,30)
  ⟨15, 3⟩,  -- [30,40)
  ⟨23, 5⟩,  -- [40,50)
  ⟨15, 9⟩,  -- [50,60)
  ⟨9, 11⟩   -- [60,70]
]

/-- Total number of surveyed customers -/
def total_surveyed : Nat := 100

/-- Expected number of shoppers on the event day -/
def expected_shoppers : Nat := 5000

/-- Theorem for the number of reusable shopping bags to prepare -/
theorem reusable_bags_estimate :
  (expected_shoppers * (survey_data.map (·.spent_more)).sum / total_surveyed : Nat) = 3500 := by
  sorry

/-- Theorem for the conditional probability -/
theorem conditional_probability_second_spender :
  let total_spent_more := (survey_data.map (·.spent_more)).sum
  let total_spent_less := (survey_data.map (·.spent_less)).sum
  (total_spent_more : Rat) / (total_surveyed - 1) = 70 / 99 := by
  sorry

end reusable_bags_estimate_conditional_probability_second_spender_l3229_322962


namespace train_crossing_time_l3229_322908

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 700 →
  train_speed = 63 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 42 := by
  sorry

end train_crossing_time_l3229_322908


namespace complement_intersection_theorem_l3229_322987

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {2, 3, 5}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3, 5} := by sorry

end complement_intersection_theorem_l3229_322987


namespace no_zero_points_for_exp_minus_x_l3229_322971

theorem no_zero_points_for_exp_minus_x :
  ∀ x : ℝ, x > 0 → ∃ ε : ℝ, ε > 0 ∧ Real.exp x - x > ε := by
  sorry

end no_zero_points_for_exp_minus_x_l3229_322971


namespace solution_f_greater_than_three_range_of_m_for_f_geq_g_l3229_322936

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := m * |x| - 2

-- Theorem for the solution of f(x) > 3
theorem solution_f_greater_than_three :
  ∀ x : ℝ, f x > 3 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for the range of m where f(x) ≥ g(x) for all x
theorem range_of_m_for_f_geq_g :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end solution_f_greater_than_three_range_of_m_for_f_geq_g_l3229_322936


namespace max_y_minus_x_l3229_322997

theorem max_y_minus_x (p q : ℕ+) (x y : ℤ) 
  (h : x * y = p * x + q * y) 
  (max_y : ∀ (y' : ℤ), x * y' = p * x + q * y' → y' ≤ y) : 
  y - x = (p - 1) * (q + 1) := by
sorry

end max_y_minus_x_l3229_322997


namespace solution_set_l3229_322952

theorem solution_set (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ∧ x + 2 > 0 → x ∈ Set.Ioo (-2) (-1) ∪ {-1} :=
by sorry

end solution_set_l3229_322952


namespace negation_equivalence_l3229_322998

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end negation_equivalence_l3229_322998


namespace max_x_value_l3229_322965

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 11) : 
  x ≤ 2 ∧ ∃ (a b : ℝ), a + b + 2 = 6 ∧ 2*a + 2*b + a*b = 11 := by
  sorry

end max_x_value_l3229_322965


namespace inequality_solution_set_l3229_322985

theorem inequality_solution_set (x : ℝ) : 
  (3 - x < x - 1) ↔ (x > 2) := by
sorry

end inequality_solution_set_l3229_322985


namespace rohans_savings_l3229_322974

/-- Rohan's monthly budget and savings calculation -/
theorem rohans_savings (salary : ℕ) (food_percent house_percent entertainment_percent conveyance_percent : ℚ) : 
  salary = 5000 →
  food_percent = 40 / 100 →
  house_percent = 20 / 100 →
  entertainment_percent = 10 / 100 →
  conveyance_percent = 10 / 100 →
  salary - (salary * (food_percent + house_percent + entertainment_percent + conveyance_percent)).floor = 1000 := by
  sorry

#check rohans_savings

end rohans_savings_l3229_322974


namespace mans_age_double_sons_l3229_322917

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 32

/-- Represents the current age of the son -/
def son_age : ℕ := 30

/-- Represents the number of years until the man's age is twice his son's age -/
def years_until_double : ℕ := 2

/-- Theorem stating that in 'years_until_double' years, the man's age will be twice his son's age -/
theorem mans_age_double_sons (y : ℕ) :
  y = years_until_double ↔ 
  (son_age + age_difference + y = 2 * (son_age + y)) :=
sorry

end mans_age_double_sons_l3229_322917


namespace equation_solution_l3229_322979

theorem equation_solution (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 := by
  sorry

end equation_solution_l3229_322979


namespace worker_usual_time_l3229_322900

/-- The usual time for a worker to reach her office, given slower speed conditions -/
theorem worker_usual_time : ∃ (T : ℝ), T = 24 ∧ T > 0 := by
  -- Let T be the usual time in minutes
  -- Let S be the usual speed in distance per minute
  -- When walking at 3/4 speed, the new time is (T + 8) minutes
  -- The distance remains constant: S * T = (3/4 * S) * (T + 8)
  sorry


end worker_usual_time_l3229_322900


namespace ptolemys_inequality_ptolemys_inequality_equality_l3229_322940

/-- Ptolemy's inequality in the complex plane -/
theorem ptolemys_inequality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) ≤ 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) :=
sorry

/-- Condition for equality in Ptolemy's inequality -/
def concyclic_or_collinear (a b c d : ℂ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - b) * (d - a)

/-- Ptolemy's inequality with equality condition -/
theorem ptolemys_inequality_equality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) = 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) ↔
  concyclic_or_collinear a b c d :=
sorry

end ptolemys_inequality_ptolemys_inequality_equality_l3229_322940


namespace wednesday_is_valid_start_day_l3229_322967

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isValidRedemptionDay (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 6, advanceDays startDay (i.val * 10) ≠ DayOfWeek.Sunday

theorem wednesday_is_valid_start_day :
  isValidRedemptionDay DayOfWeek.Wednesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Wednesday → ¬isValidRedemptionDay d :=
sorry

end wednesday_is_valid_start_day_l3229_322967


namespace solution_set_when_a_eq_2_range_of_a_l3229_322912

-- Define the functions f and g
def f (a x : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Part 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ≥ 2 :=
sorry

end solution_set_when_a_eq_2_range_of_a_l3229_322912


namespace det_A_plus_5_l3229_322942

def A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 7]

theorem det_A_plus_5 : Matrix.det A + 5 = 41 := by
  sorry

end det_A_plus_5_l3229_322942


namespace circle_tangent_to_parabola_directrix_l3229_322986

/-- Given a circle and a parabola, if the circle is tangent to the directrix of the parabola,
    then the parameter p of the parabola is 2. -/
theorem circle_tangent_to_parabola_directrix (x y : ℝ) (p : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (p > 0) →                   -- p is positive
  (∃ (y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (x₀ : ℝ), ∀ (x y : ℝ), x^2 + y^2 - 6*x - 7 = 0 → |x - x₀| ≥ p/2) →  -- Circle is tangent to directrix
  p = 2 := by
sorry

end circle_tangent_to_parabola_directrix_l3229_322986


namespace josie_remaining_money_l3229_322918

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific initial amount and purchase costs, the remaining amount is correct -/
theorem josie_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end josie_remaining_money_l3229_322918


namespace total_population_is_56000_l3229_322976

/-- The total population of Boise, Seattle, and Lake View -/
def total_population (boise seattle lakeview : ℕ) : ℕ :=
  boise + seattle + lakeview

/-- Theorem: The total population of the three cities is 56000 -/
theorem total_population_is_56000 :
  ∃ (boise seattle lakeview : ℕ),
    boise = (3 * seattle) / 5 ∧
    lakeview = seattle + 4000 ∧
    lakeview = 24000 ∧
    total_population boise seattle lakeview = 56000 := by
  sorry

end total_population_is_56000_l3229_322976


namespace inscribed_square_side_length_l3229_322929

/-- A regular triangle with an inscribed square -/
structure RegularTriangleWithInscribedSquare where
  -- Side length of the regular triangle
  triangleSide : ℝ
  -- Distance from a vertex of the triangle to the nearest vertex of the square on the opposite side
  vertexToSquareDistance : ℝ
  -- Assumption that the triangle is regular (equilateral)
  regular : triangleSide > 0
  -- Assumption that the square is inscribed (vertexToSquareDistance < triangleSide)
  inscribed : vertexToSquareDistance < triangleSide

/-- The side length of the inscribed square -/
def squareSideLength (t : RegularTriangleWithInscribedSquare) : ℝ := 
  t.triangleSide - t.vertexToSquareDistance

/-- Theorem stating that for a regular triangle with side length 30 and vertexToSquareDistance 29, 
    the side length of the inscribed square is 30 -/
theorem inscribed_square_side_length 
  (t : RegularTriangleWithInscribedSquare) 
  (h1 : t.triangleSide = 30) 
  (h2 : t.vertexToSquareDistance = 29) : 
  squareSideLength t = 30 := by
  sorry

end inscribed_square_side_length_l3229_322929


namespace product_divisible_by_nine_l3229_322902

theorem product_divisible_by_nine : ∃ k : ℤ, 12345 * 54321 = 9 * k := by
  sorry

end product_divisible_by_nine_l3229_322902


namespace no_real_solutions_for_equation_l3229_322988

theorem no_real_solutions_for_equation : ¬∃ x : ℝ, x + Real.sqrt (x - 2) = 6 := by
  sorry

end no_real_solutions_for_equation_l3229_322988


namespace circumscribed_trapezoid_radius_l3229_322943

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The lateral side of the trapezoid -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The height is half the lateral side -/
  height_half_lateral : height = lateral_side / 2
  /-- The area is positive -/
  area_pos : 0 < area

/-- 
  For an isosceles trapezoid circumscribed around a circle, 
  if the area of the trapezoid is S and its height is half of its lateral side, 
  then the radius of the circle is √(S/8).
-/
theorem circumscribed_trapezoid_radius 
  (t : CircumscribedTrapezoid) : t.radius = Real.sqrt (t.area / 8) := by
  sorry

end circumscribed_trapezoid_radius_l3229_322943


namespace custom_set_op_theorem_l3229_322983

-- Define the custom set operation ⊗
def customSetOp (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem custom_set_op_theorem :
  customSetOp M N = {x | (-2 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

end custom_set_op_theorem_l3229_322983


namespace ae_bc_ratio_l3229_322939

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the triangles
def is_equilateral (X Y Z : ℝ × ℝ) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Define the midpoint
def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem ae_bc_ratio (A B C D E : ℝ × ℝ) :
  is_equilateral A B C →
  is_equilateral B C D →
  is_equilateral C D E →
  is_midpoint E C D →
  dist A E / dist B C = Real.sqrt 3 := by
  sorry

end ae_bc_ratio_l3229_322939


namespace janet_extra_fica_tax_l3229_322901

/-- Represents Janet's employment situation -/
structure Employment where
  hours_per_week : ℕ
  current_hourly_rate : ℚ
  freelance_hourly_rate : ℚ
  healthcare_premium_per_month : ℚ
  additional_monthly_income_freelancing : ℚ

/-- Calculates the extra weekly FICA tax for freelancing -/
def extra_weekly_fica_tax (e : Employment) : ℚ :=
  let current_monthly_income := e.hours_per_week * e.current_hourly_rate * 4
  let freelance_monthly_income := e.hours_per_week * e.freelance_hourly_rate * 4
  let extra_monthly_income := freelance_monthly_income - current_monthly_income
  let extra_monthly_income_after_healthcare := extra_monthly_income - e.healthcare_premium_per_month
  (extra_monthly_income_after_healthcare - e.additional_monthly_income_freelancing) / 4

/-- Theorem stating that the extra weekly FICA tax for Janet's situation is $25 -/
theorem janet_extra_fica_tax :
  let janet : Employment := {
    hours_per_week := 40,
    current_hourly_rate := 30,
    freelance_hourly_rate := 40,
    healthcare_premium_per_month := 400,
    additional_monthly_income_freelancing := 1100
  }
  extra_weekly_fica_tax janet = 25 := by sorry

end janet_extra_fica_tax_l3229_322901


namespace milk_container_problem_l3229_322972

theorem milk_container_problem (x : ℝ) : 
  (3 * x + 2 * 0.75 + 5 * 0.5 = 10) → x = 2 := by
  sorry

end milk_container_problem_l3229_322972


namespace f_increasing_interval_f_greater_than_linear_l3229_322938

noncomputable section

def f (x : ℝ) := Real.log x - (1/2) * (x - 1)^2

theorem f_increasing_interval (x : ℝ) (hx : x > 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = Real.sqrt 2 ∧
  ∀ y z, a < y ∧ y < z ∧ z < b → f y < f z :=
sorry

theorem f_greater_than_linear (k : ℝ) :
  (∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ↔ k < 1 :=
sorry

end f_increasing_interval_f_greater_than_linear_l3229_322938


namespace solution_set_f_leq_6_range_of_a_l3229_322970

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem 1: Solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Range of a for f(x) ≥ |2x+a| - 4 when x ∈ [-1/2, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/2) 1, f x ≥ |2*x + a| - 4) ↔ -7 ≤ a ∧ a ≤ 6 := by sorry

end solution_set_f_leq_6_range_of_a_l3229_322970


namespace find_k_angle_90_degrees_l3229_322995

-- Define vectors in R^2
def a : Fin 2 → ℝ := ![3, -1]
def b (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define dot product for 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Theorem 1: Find the value of k
theorem find_k : ∃ k : ℝ, perpendicular a (b k) ∧ k = 3 := by sorry

-- Define vector addition and subtraction
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 + w 0, v 1 + w 1]
def sub_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 - w 0, v 1 - w 1]

-- Theorem 2: Prove the angle between a + b and a - b is 90°
theorem angle_90_degrees : 
  let b' := b 3
  let sum := add_vectors a b'
  let diff := sub_vectors a b'
  perpendicular sum diff := by sorry

end find_k_angle_90_degrees_l3229_322995


namespace total_investment_equals_eight_thousand_l3229_322932

/-- Represents an investment account with a given balance and interest rate. -/
structure Account where
  balance : ℝ
  interestRate : ℝ

/-- Calculates the total investment given two accounts. -/
def totalInvestment (account1 account2 : Account) : ℝ :=
  account1.balance + account2.balance

/-- Theorem: The total investment in two accounts with $4,000 each is $8,000. -/
theorem total_investment_equals_eight_thousand 
  (account1 account2 : Account)
  (h1 : account1.balance = 4000)
  (h2 : account2.balance = 4000) :
  totalInvestment account1 account2 = 8000 := by
  sorry

#check total_investment_equals_eight_thousand

end total_investment_equals_eight_thousand_l3229_322932


namespace division_problem_l3229_322977

theorem division_problem (quotient divisor remainder : ℕ) 
  (h1 : quotient = 3)
  (h2 : divisor = 3)
  (h3 : divisor = 3 * remainder) : 
  quotient * divisor + remainder = 10 := by
sorry

end division_problem_l3229_322977


namespace inequality_equivalence_fraction_comparison_l3229_322947

-- Problem 1
theorem inequality_equivalence (m x : ℝ) (h : m > 2) :
  m * x + 4 < m^2 + 2 * x ↔ x < m + 2 := by sorry

-- Problem 2
theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  x / (1 + x) > y / (1 + y) := by sorry

end inequality_equivalence_fraction_comparison_l3229_322947


namespace solution_set_quadratic_inequality_l3229_322984

-- Define the quadratic equation and its roots
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the sets S, T, P, Q
def S (x₁ : ℝ) : Set ℝ := {x | x > x₁}
def T (x₂ : ℝ) : Set ℝ := {x | x > x₂}
def P (x₁ : ℝ) : Set ℝ := {x | x < x₁}
def Q (x₂ : ℝ) : Set ℝ := {x | x < x₂}

-- State the theorem
theorem solution_set_quadratic_inequality 
  (a b c x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation a b c x₁)
  (h₂ : quadratic_equation a b c x₂)
  (h₃ : x₁ ≠ x₂)
  (h₄ : a > 0) :
  {x : ℝ | a * x^2 + b * x + c > 0} = (S x₁ ∩ T x₂) ∪ (P x₁ ∩ Q x₂) := by
  sorry

end solution_set_quadratic_inequality_l3229_322984


namespace least_k_value_l3229_322925

/-- The function f(t) = t² - t + 1 -/
def f (t : ℝ) : ℝ := t^2 - t + 1

/-- The property that needs to be satisfied for all x, y, z that are not all positive -/
def satisfies_property (k : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬(x > 0 ∧ y > 0 ∧ z > 0) →
    k * f x * f y * f z ≥ f (x * y * z)

/-- The theorem stating that 16/9 is the least value of k satisfying the property -/
theorem least_k_value : 
  (∀ k : ℝ, k < 16/9 → ¬(satisfies_property k)) ∧ 
  satisfies_property (16/9) := by sorry

end least_k_value_l3229_322925


namespace derivative_at_one_implies_a_value_l3229_322953

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a) * log x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (deriv (f a)) 1 = -2 → a = 3 := by
  sorry

end derivative_at_one_implies_a_value_l3229_322953


namespace notebooks_in_class2_l3229_322951

/-- The number of notebooks that do not belong to Class 1 -/
def not_class1 : ℕ := 162

/-- The number of notebooks that do not belong to Class 2 -/
def not_class2 : ℕ := 143

/-- The number of notebooks that belong to both Class 1 and Class 2 -/
def both_classes : ℕ := 87

/-- The total number of notebooks -/
def total_notebooks : ℕ := not_class1 + not_class2 - both_classes

theorem notebooks_in_class2 : 
  total_notebooks - (total_notebooks - not_class2) = 53 := by
  sorry

end notebooks_in_class2_l3229_322951


namespace broken_eggs_count_l3229_322973

/-- Given a total of 24 eggs, where some are broken, some are cracked, and some are perfect,
    prove that the number of broken eggs is 3 under the following conditions:
    1. The number of cracked eggs is twice the number of broken eggs
    2. The difference between perfect and cracked eggs is 9 -/
theorem broken_eggs_count (broken : ℕ) (cracked : ℕ) (perfect : ℕ) : 
  perfect + cracked + broken = 24 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken = 3 := by sorry

end broken_eggs_count_l3229_322973
