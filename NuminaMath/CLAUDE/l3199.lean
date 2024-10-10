import Mathlib

namespace area_covered_by_specific_strips_l3199_319952

/-- Calculates the area covered by four rectangular strips on a table. -/
def areaCoveredByStrips (lengths : List Nat) (width : Nat) (overlaps : Nat) : Nat :=
  let totalArea := (lengths.sum * width)
  let overlapArea := overlaps * width
  totalArea - overlapArea

/-- Theorem: The area covered by four specific strips is 33. -/
theorem area_covered_by_specific_strips :
  areaCoveredByStrips [12, 10, 8, 6] 1 3 = 33 := by
  sorry

end area_covered_by_specific_strips_l3199_319952


namespace min_broken_line_length_l3199_319974

/-- Given points A and C in the coordinate plane, and point B on the x-axis,
    the minimum length of the broken line ABC is 7.5 -/
theorem min_broken_line_length :
  let A : ℝ × ℝ := (-3, -4)
  let C : ℝ × ℝ := (1.5, -2)
  ∃ B : ℝ × ℝ, B.2 = 0 ∧
    ∀ B' : ℝ × ℝ, B'.2 = 0 →
      Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
      Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≤
      Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2) +
      Real.sqrt ((C.1 - B'.1)^2 + (C.2 - B'.2)^2) ∧
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 7.5 := by
  sorry

end min_broken_line_length_l3199_319974


namespace square_over_fraction_equals_324_l3199_319981

theorem square_over_fraction_equals_324 : (45^2 : ℚ) / (7 - 3/4) = 324 := by
  sorry

end square_over_fraction_equals_324_l3199_319981


namespace jess_walk_distance_l3199_319904

/-- The number of blocks Jess must walk to complete her errands and arrive at work -/
def remaining_blocks (post_office store gallery library work already_walked : ℕ) : ℕ :=
  post_office + store + gallery + library + work - already_walked

/-- Theorem stating the number of blocks Jess must walk given the problem conditions -/
theorem jess_walk_distance :
  remaining_blocks 24 18 15 14 22 9 = 84 := by
  sorry

end jess_walk_distance_l3199_319904


namespace trigonometric_expression_equals_four_l3199_319997

theorem trigonometric_expression_equals_four : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
sorry

end trigonometric_expression_equals_four_l3199_319997


namespace geometric_sequence_150th_term_l3199_319908

/-- Given a geometric sequence with first term 8 and second term -4, 
    the 150th term is equal to -8 * (1/2)^149 -/
theorem geometric_sequence_150th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (-1/2)) → 
    a 1 = 8 → 
    a 2 = -4 → 
    a 150 = -8 * (1/2)^149 := by
  sorry

end geometric_sequence_150th_term_l3199_319908


namespace calc_1_calc_2_calc_3_calc_4_l3199_319985

-- (1) 327 + 46 - 135 = 238
theorem calc_1 : 327 + 46 - 135 = 238 := by sorry

-- (2) 1000 - 582 - 128 = 290
theorem calc_2 : 1000 - 582 - 128 = 290 := by sorry

-- (3) (124 - 62) × 6 = 372
theorem calc_3 : (124 - 62) * 6 = 372 := by sorry

-- (4) 500 - 400 ÷ 5 = 420
theorem calc_4 : 500 - 400 / 5 = 420 := by sorry

end calc_1_calc_2_calc_3_calc_4_l3199_319985


namespace exist_five_integers_sum_four_is_square_l3199_319990

theorem exist_five_integers_sum_four_is_square : ∃ (a₁ a₂ a₃ a₄ a₅ : ℤ),
  (a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₄ ≠ a₅) ∧
  (∃ n₁ : ℕ, a₂ + a₃ + a₄ + a₅ = n₁^2) ∧
  (∃ n₂ : ℕ, a₁ + a₃ + a₄ + a₅ = n₂^2) ∧
  (∃ n₃ : ℕ, a₁ + a₂ + a₄ + a₅ = n₃^2) ∧
  (∃ n₄ : ℕ, a₁ + a₂ + a₃ + a₅ = n₄^2) ∧
  (∃ n₅ : ℕ, a₁ + a₂ + a₃ + a₄ = n₅^2) :=
by sorry

end exist_five_integers_sum_four_is_square_l3199_319990


namespace line_intercept_sum_l3199_319961

/-- Given a line mx + 3y - 12 = 0 where m is a real number,
    if the sum of its intercepts on the x and y axes is 7,
    then m = 4. -/
theorem line_intercept_sum (m : ℝ) : 
  (∃ x y : ℝ, m * x + 3 * y - 12 = 0 ∧ 
   (x = 0 ∨ y = 0) ∧
   (∃ x₀ y₀ : ℝ, m * x₀ + 3 * y₀ - 12 = 0 ∧ 
    x₀ = 0 ∧ y₀ = 0 ∧ x + y₀ = 7)) → 
  m = 4 := by
sorry

end line_intercept_sum_l3199_319961


namespace normal_distribution_symmetry_l3199_319944

/-- A random vector following a normal distribution with mean 3 and variance 1 -/
def X : Type := Real

/-- The probability density function of X -/
noncomputable def pdf (x : X) : Real := sorry

/-- The cumulative distribution function of X -/
noncomputable def cdf (x : X) : Real := sorry

/-- The probability that X is greater than a given value -/
noncomputable def P_greater (a : Real) : Real := 1 - cdf a

/-- The probability that X is less than a given value -/
noncomputable def P_less (a : Real) : Real := cdf a

/-- The theorem stating that if P(X > 2c - 1) = P(X < c + 3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : Real) :
  P_greater (2 * c - 1) = P_less (c + 3) → c = 4/3 := by sorry

end normal_distribution_symmetry_l3199_319944


namespace intersection_point_of_function_and_inverse_l3199_319987

theorem intersection_point_of_function_and_inverse
  (b a : ℤ) (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (h1 : ∀ x, g x = 4 * x + b)
  (h2 : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g)
  (h3 : g (-4) = a)
  (h4 : g_inv (-4) = a) :
  a = -4 := by
  sorry

end intersection_point_of_function_and_inverse_l3199_319987


namespace perfect_square_power_of_two_l3199_319999

theorem perfect_square_power_of_two (n : ℕ) : 
  (∃ m : ℕ, 2^5 + 2^11 + 2^n = m^2) ↔ n = 12 := by
sorry

end perfect_square_power_of_two_l3199_319999


namespace remaining_pets_count_l3199_319976

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : ℕ
  kittens : ℕ
  rabbits : ℕ
  guineaPigs : ℕ
  chameleons : ℕ
  parrots : ℕ

/-- Calculates the total number of pets -/
def totalPets (counts : PetCounts) : ℕ :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guineaPigs + counts.chameleons + counts.parrots

/-- Represents the pet store transactions throughout the day -/
def petStoreTransactions (initial : PetCounts) : PetCounts :=
  { puppies := initial.puppies - 2 - 1 + 3 - 1 - 1,
    kittens := initial.kittens - 1 - 2 + 2 - 1 + 1 - 1,
    rabbits := initial.rabbits - 1 - 1 + 1 - 1 - 1,
    guineaPigs := initial.guineaPigs - 1 - 2 - 1 - 1,
    chameleons := initial.chameleons + 1 + 2 - 1,
    parrots := initial.parrots - 1 }

/-- The main theorem stating that after all transactions, 16 pets remain -/
theorem remaining_pets_count (initial : PetCounts)
    (h_initial : initial = { puppies := 7, kittens := 6, rabbits := 4,
                             guineaPigs := 5, chameleons := 3, parrots := 2 }) :
    totalPets (petStoreTransactions initial) = 16 := by
  sorry


end remaining_pets_count_l3199_319976


namespace negation_of_proposition_P_l3199_319996

theorem negation_of_proposition_P :
  (¬ (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_proposition_P_l3199_319996


namespace intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l3199_319975

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for complement of A ∪ B in U
theorem complement_union_A_B : (A ∪ B)ᶜ = {x | x < -3 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for (complement of A in U) ∪ B
theorem complement_A_union_B : Aᶜ ∪ B = {x | x ≤ 2 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for A ∩ (complement of B in U)
theorem A_intersection_complement_B : A ∩ Bᶜ = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for (complement of A in U) ∪ (complement of B in U)
theorem complement_A_union_complement_B : Aᶜ ∪ Bᶜ = {x | x ≤ -2 ∨ 2 < x} ∩ U := by sorry

end intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l3199_319975


namespace function_maximum_value_l3199_319920

theorem function_maximum_value (x : ℝ) (h : x > 4) :
  (fun x => -x + 1 / (4 - x)) x ≤ -6 :=
by sorry

end function_maximum_value_l3199_319920


namespace arithmetic_mean_problem_l3199_319979

theorem arithmetic_mean_problem : ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 9 ∧ x = 13 := by
  sorry

end arithmetic_mean_problem_l3199_319979


namespace normal_vector_of_det_equation_l3199_319935

/-- The determinant equation of a line -/
def det_equation (x y : ℝ) : Prop := x * 1 - y * 2 = 0

/-- Definition of a normal vector -/
def is_normal_vector (n : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line_eq x y → n.1 * x + n.2 * y = 0

/-- Theorem: (1, -2) is a normal vector of the line represented by |x 2; y 1| = 0 -/
theorem normal_vector_of_det_equation :
  is_normal_vector (1, -2) det_equation :=
sorry

end normal_vector_of_det_equation_l3199_319935


namespace sufficient_not_necessary_l3199_319901

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line l: x + y - 1 = 0 -/
def line_l (p : Point) : Prop := p.x + p.y - 1 = 0

/-- The specific condition x=2 and y=-1 -/
def specific_condition (p : Point) : Prop := p.x = 2 ∧ p.y = -1

/-- Theorem stating that the specific condition is sufficient but not necessary -/
theorem sufficient_not_necessary :
  (∀ p : Point, specific_condition p → line_l p) ∧
  ¬(∀ p : Point, line_l p → specific_condition p) := by
  sorry

end sufficient_not_necessary_l3199_319901


namespace set_B_equals_l3199_319950

def A : Set ℝ := {x | x^2 ≤ 4}

def B : Set ℕ := {x | x > 0 ∧ (x - 1 : ℝ) ∈ A}

theorem set_B_equals : B = {1, 2, 3} := by sorry

end set_B_equals_l3199_319950


namespace cricketer_average_score_l3199_319919

theorem cricketer_average_score (total_matches : ℕ) (first_matches : ℕ) (last_matches : ℕ)
  (first_avg : ℚ) (last_avg : ℚ) :
  total_matches = first_matches + last_matches →
  total_matches = 12 →
  first_matches = 8 →
  last_matches = 4 →
  first_avg = 40 →
  last_avg = 64 →
  (first_avg * first_matches + last_avg * last_matches) / total_matches = 48 := by
  sorry

#check cricketer_average_score

end cricketer_average_score_l3199_319919


namespace unique_solution_equation_l3199_319988

theorem unique_solution_equation :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (a * b + 1) :=
by
  sorry

end unique_solution_equation_l3199_319988


namespace baseball_team_wins_l3199_319930

theorem baseball_team_wins (total_games : ℕ) (ratio : ℚ) (wins : ℕ) : 
  total_games = 10 → 
  ratio = 2 → 
  ratio = total_games / (total_games - wins) → 
  wins = 5 := by
sorry

end baseball_team_wins_l3199_319930


namespace mexican_food_pricing_l3199_319970

/-- Given the pricing conditions for Mexican food items, prove the cost of a specific combination. -/
theorem mexican_food_pricing
  (enchilada_price taco_price burrito_price : ℚ)
  (h1 : 2 * enchilada_price + 3 * taco_price + burrito_price = 5)
  (h2 : 3 * enchilada_price + 2 * taco_price + 2 * burrito_price = 15/2) :
  2 * enchilada_price + 2 * taco_price + 3 * burrito_price = 85/8 := by
  sorry

end mexican_food_pricing_l3199_319970


namespace apples_eaten_per_day_l3199_319991

theorem apples_eaten_per_day 
  (initial_apples : ℕ) 
  (remaining_apples : ℕ) 
  (days : ℕ) 
  (h1 : initial_apples = 32) 
  (h2 : remaining_apples = 4) 
  (h3 : days = 7) :
  (initial_apples - remaining_apples) / days = 4 :=
by sorry

end apples_eaten_per_day_l3199_319991


namespace kristy_work_hours_l3199_319941

/-- Proves that given the conditions of Kristy's salary structure and earnings,
    she worked 160 hours in the month. -/
theorem kristy_work_hours :
  let hourly_rate : ℝ := 7.5
  let commission_rate : ℝ := 0.16
  let total_sales : ℝ := 25000
  let insurance_amount : ℝ := 260
  let insurance_rate : ℝ := 0.05
  let commission : ℝ := commission_rate * total_sales
  let total_earnings : ℝ := insurance_amount / insurance_rate
  let hours_worked : ℝ := (total_earnings - commission) / hourly_rate
  hours_worked = 160 := by sorry

end kristy_work_hours_l3199_319941


namespace multiplicative_inverse_289_mod_391_l3199_319966

theorem multiplicative_inverse_289_mod_391 
  (h : 136^2 + 255^2 = 289^2) : 
  (289 * 18) % 391 = 1 ∧ 0 ≤ 18 ∧ 18 < 391 := by
  sorry

end multiplicative_inverse_289_mod_391_l3199_319966


namespace car_travel_distance_l3199_319953

theorem car_travel_distance (speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) (distance : ℝ) : 
  speed = 80 →
  speed_increase = 40 →
  time_decrease = 0.5 →
  distance / speed - distance / (speed + speed_increase) = time_decrease →
  distance = 120 := by
  sorry

#check car_travel_distance

end car_travel_distance_l3199_319953


namespace lenas_collage_friends_l3199_319959

/-- Given the conditions of Lena's collage project, prove the number of friends' pictures glued. -/
theorem lenas_collage_friends (clippings_per_friend : ℕ) (glue_per_clipping : ℕ) (total_glue : ℕ) 
  (h1 : clippings_per_friend = 3)
  (h2 : glue_per_clipping = 6)
  (h3 : total_glue = 126) :
  total_glue / (clippings_per_friend * glue_per_clipping) = 7 := by
  sorry

end lenas_collage_friends_l3199_319959


namespace special_subset_contains_all_rationals_l3199_319907

def is_special_subset (S : Set ℚ) : Prop :=
  (1/2 ∈ S) ∧ 
  (∀ x ∈ S, x/2 ∈ S) ∧ 
  (∀ x ∈ S, 1/(x+1) ∈ S)

theorem special_subset_contains_all_rationals (S : Set ℚ) 
  (h : is_special_subset S) :
  ∀ q ∈ Set.Ioo (0 : ℚ) 1, q ∈ S :=
by
  sorry

end special_subset_contains_all_rationals_l3199_319907


namespace local_min_condition_l3199_319912

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a^2 - 2*a)*x

/-- Theorem stating the condition for x = 0 to be a local minimum of f -/
theorem local_min_condition (a : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → f a x ≥ f a 0) ↔ a < 0 ∨ a > 2 :=
sorry

end local_min_condition_l3199_319912


namespace divide_into_eight_parts_l3199_319954

-- Define a type for geometric figures
inductive Figure
  | Cube
  | Rectangle

-- Define a function to check if a figure can be divided into 8 identical parts
def canDivideIntoEightParts (f : Figure) : Prop :=
  match f with
  | Figure.Cube => true
  | Figure.Rectangle => true

-- Theorem stating that any cube or rectangle can be divided into 8 identical parts
theorem divide_into_eight_parts (f : Figure) : canDivideIntoEightParts f := by
  sorry

#check divide_into_eight_parts

end divide_into_eight_parts_l3199_319954


namespace calculation_proofs_l3199_319921

theorem calculation_proofs :
  (6 / (-1/2 + 1/3) = -36) ∧
  ((-14/17) * 99 + (13/17) * 99 - (16/17) * 99 = -99) := by
  sorry

end calculation_proofs_l3199_319921


namespace hawkeye_battery_charge_cost_l3199_319917

theorem hawkeye_battery_charge_cost 
  (budget : ℝ) 
  (num_charges : ℕ) 
  (remaining : ℝ) 
  (h1 : budget = 20)
  (h2 : num_charges = 4)
  (h3 : remaining = 6) : 
  (budget - remaining) / num_charges = 3.50 := by
  sorry

end hawkeye_battery_charge_cost_l3199_319917


namespace wednesday_temperature_l3199_319939

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C under the given conditions. -/
theorem wednesday_temperature (M T W : ℤ) : 
  T = M + 4 →   -- Tuesday's temperature is 4°C warmer than Monday's
  W = M - 6 →   -- Wednesday's temperature is 6°C cooler than Monday's
  T = 22 →      -- Tuesday's temperature is 22°C
  W = 12 :=     -- Prove: Wednesday's temperature is 12°C
by sorry

end wednesday_temperature_l3199_319939


namespace jimmy_shorts_count_l3199_319913

def senior_discount : ℚ := 10 / 100
def shorts_price : ℚ := 15
def shirt_price : ℚ := 17
def num_shirts : ℕ := 5
def total_paid : ℚ := 117

def num_shorts : ℕ := 2

theorem jimmy_shorts_count :
  let shirts_cost := shirt_price * num_shirts
  let discount := shirts_cost * senior_discount
  let irene_total := shirts_cost - discount
  let remaining := total_paid - irene_total
  (remaining / shorts_price).floor = num_shorts := by sorry

end jimmy_shorts_count_l3199_319913


namespace area_is_100_l3199_319945

/-- The area enclosed by the graph of |x| + |2y| = 10 -/
def area_enclosed : ℝ := 100

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs x + abs (2 * y) = 10

/-- The graph is symmetric across the x-axis -/
axiom symmetric_x_axis : ∀ x y : ℝ, graph_equation x y → graph_equation x (-y)

/-- The graph is symmetric across the y-axis -/
axiom symmetric_y_axis : ∀ x y : ℝ, graph_equation x y → graph_equation (-x) y

/-- The graph forms four congruent triangles -/
axiom four_congruent_triangles : ∃ A : ℝ, A > 0 ∧ area_enclosed = 4 * A

/-- Theorem: The area enclosed by the graph of |x| + |2y| = 10 is 100 square units -/
theorem area_is_100 : area_enclosed = 100 := by sorry

end area_is_100_l3199_319945


namespace quadrilateral_area_rational_l3199_319927

/-- The area of a quadrilateral with integer coordinates is rational -/
theorem quadrilateral_area_rational
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ) :
  ∃ (q : ℚ), q = |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)| / 2 +
              |x₁ * (y₃ - y₄) + x₃ * (y₄ - y₁) + x₄ * (y₁ - y₃)| / 2 :=
by sorry

end quadrilateral_area_rational_l3199_319927


namespace least_repeating_digits_7_13_is_6_l3199_319943

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_digits_7_13 : ℕ :=
  6

/-- Theorem stating that the least number of digits in a repeating block 
    of the decimal expansion of 7/13 is 6 -/
theorem least_repeating_digits_7_13_is_6 :
  least_repeating_digits_7_13 = 6 := by sorry

end least_repeating_digits_7_13_is_6_l3199_319943


namespace power_calculation_l3199_319924

theorem power_calculation : (8^5 / 8^2) * 2^10 / 2^3 = 65536 := by
  sorry

end power_calculation_l3199_319924


namespace distance_between_points_l3199_319936

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 18)
  let p2 : ℝ × ℝ := (13, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 269 := by
sorry

end distance_between_points_l3199_319936


namespace inequality_proof_l3199_319980

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end inequality_proof_l3199_319980


namespace right_triangle_pq_length_l3199_319957

/-- Given a right triangle PQR with ∠P = 90°, QR = 15, and tan R = 5 cos Q, prove that PQ = 6√6 -/
theorem right_triangle_pq_length (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let cos_q := pq / qr
  let tan_r := pq / pr
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at P
  qr = 15 →
  tan_r = 5 * cos_q →
  pq = 6 * Real.sqrt 6 := by
sorry


end right_triangle_pq_length_l3199_319957


namespace circle_condition_tangent_lines_perpendicular_intersection_l3199_319940

-- Define the equation of circle C
def C (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + a = 0

-- Define the line l
def l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem 1: C represents a circle iff a ∈ (-∞, 8)
theorem circle_condition (a : ℝ) : 
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), C x y a ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ 
  a < 8 :=
sorry

-- Theorem 2: Tangent lines when a = -17
theorem tangent_lines : 
  (∀ (x y : ℝ), C x y (-17) → (39*x + 80*y - 207 = 0 ∨ x = 7)) ∧
  C 7 (-6) (-17) ∧
  (∃ (x y : ℝ), C x y (-17) ∧ 39*x + 80*y - 207 = 0) ∧
  (∃ (y : ℝ), C 7 y (-17)) :=
sorry

-- Theorem 3: Value of a when OA ⊥ OB
theorem perpendicular_intersection :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C x₁ y₁ (-6/5) ∧ C x₂ y₂ (-6/5) ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) :=
sorry

end circle_condition_tangent_lines_perpendicular_intersection_l3199_319940


namespace last_four_digits_of_perfect_square_l3199_319906

theorem last_four_digits_of_perfect_square (n : ℕ) : 
  (∃ d : ℕ, d < 10 ∧ n^2 % 1000 = d * 111) → 
  (n^2 % 10000 = 0 ∨ n^2 % 10000 = 1444) :=
sorry

end last_four_digits_of_perfect_square_l3199_319906


namespace enrollment_difference_l3199_319971

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : ℕ

/-- Theorem: The positive difference between the maximum and minimum enrollments is 750 --/
theorem enrollment_difference (schools : List School) 
  (h1 : schools.length = 5)
  (h2 : ∃ s ∈ schools, s.name = "Varsity" ∧ s.enrollment = 1680)
  (h3 : ∃ s ∈ schools, s.name = "Northwest" ∧ s.enrollment = 1170)
  (h4 : ∃ s ∈ schools, s.name = "Central" ∧ s.enrollment = 1840)
  (h5 : ∃ s ∈ schools, s.name = "Greenbriar" ∧ s.enrollment = 1090)
  (h6 : ∃ s ∈ schools, s.name = "Eastside" ∧ s.enrollment = 1450) :
  (schools.map (·.enrollment)).maximum?.get! - (schools.map (·.enrollment)).minimum?.get! = 750 := by
  sorry


end enrollment_difference_l3199_319971


namespace doubled_b_cost_percentage_l3199_319993

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_b_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h' : b > 0) :
  cost t (2*b) = 16 * cost t b := by
  sorry

end doubled_b_cost_percentage_l3199_319993


namespace leah_peeled_18_l3199_319967

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  leah_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Leah peeled -/
def leah_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_left := scenario.total_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.leah_rate
  let combined_time := potatoes_left / combined_rate
  scenario.leah_rate * combined_time

/-- The theorem stating that Leah peeled 18 potatoes -/
theorem leah_peeled_18 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 50)
  (h2 : scenario.homer_rate = 3)
  (h3 : scenario.leah_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  leah_potatoes scenario = 18 := by
  sorry

end leah_peeled_18_l3199_319967


namespace triangle_formation_l3199_319938

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (5, 6, 10), (1, 2, 3)]

theorem triangle_formation :
  ∃! (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c :=
sorry

end triangle_formation_l3199_319938


namespace least_perimeter_of_triangle_l3199_319923

theorem least_perimeter_of_triangle (a b c : ℕ) : 
  a = 24 → b = 51 → c > 0 → a + b > c → a + c > b → b + c > a → 
  ∀ x : ℕ, (x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a) → a + b + c ≤ a + b + x →
  a + b + c = 103 := by sorry

end least_perimeter_of_triangle_l3199_319923


namespace pipe_ratio_proof_l3199_319914

theorem pipe_ratio_proof (total_length longer_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : longer_length = 118)
  (h3 : ∃ k : ℕ, k * (total_length - longer_length) = longer_length) :
  longer_length / (total_length - longer_length) = 2 :=
by
  sorry

end pipe_ratio_proof_l3199_319914


namespace sum_of_three_numbers_l3199_319978

theorem sum_of_three_numbers (p q r M : ℚ) 
  (h1 : p + q + r = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : r / 5 = M) :
  M = 15 := by
  sorry

end sum_of_three_numbers_l3199_319978


namespace smallest_k_for_divisible_sum_of_squares_l3199_319992

/-- The sum of squares from 1 to n -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is divisible by 150 -/
def divisibleBy150 (n : ℕ) : Prop := ∃ m : ℕ, n = 150 * m

theorem smallest_k_for_divisible_sum_of_squares :
  (∀ k : ℕ, 0 < k ∧ k < 100 → ¬(divisibleBy150 (sumOfSquares k))) ∧
  (divisibleBy150 (sumOfSquares 100)) := by
  sorry

#check smallest_k_for_divisible_sum_of_squares

end smallest_k_for_divisible_sum_of_squares_l3199_319992


namespace inverse_proportion_ratio_l3199_319916

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_inverse : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h_nonzero_x : x₁ ≠ 0 ∧ x₂ ≠ 0) (h_nonzero_y : y₁ ≠ 0 ∧ y₂ ≠ 0) (h_ratio_x : x₁ / x₂ = 4 / 5) 
    (h_correspond : x₁ * y₁ = x₂ * y₂) : y₁ / y₂ = 5 / 4 := by
  sorry

end inverse_proportion_ratio_l3199_319916


namespace smallest_k_for_triangular_l3199_319909

/-- A positive integer T is triangular if there exists a positive integer n such that T = n * (n + 1) / 2 -/
def IsTriangular (T : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ T = n * (n + 1) / 2

/-- The smallest positive integer k such that for any triangular number T, 81T + k is also triangular -/
theorem smallest_k_for_triangular : ∃! k : ℕ, 
  k > 0 ∧ 
  (∀ T : ℕ, IsTriangular T → IsTriangular (81 * T + k)) ∧
  (∀ k' : ℕ, k' > 0 → k' < k → 
    ∃ T : ℕ, IsTriangular T ∧ ¬IsTriangular (81 * T + k')) :=
sorry

end smallest_k_for_triangular_l3199_319909


namespace ceiling_neg_sqrt_64_9_l3199_319949

theorem ceiling_neg_sqrt_64_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_9_l3199_319949


namespace solve_equation_l3199_319934

theorem solve_equation (x : ℝ) : 1 - 2 / (1 + x) = 1 / (1 + x) → x = 2 := by
  sorry

end solve_equation_l3199_319934


namespace two_roots_sum_greater_than_2a_l3199_319929

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 2

theorem two_roots_sum_greater_than_2a (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ > 2 * a := by
  sorry

end two_roots_sum_greater_than_2a_l3199_319929


namespace power_of_five_congruences_and_digits_l3199_319900

theorem power_of_five_congruences_and_digits : 
  (∃ k : ℕ, 5^500 = 1000 * k + 1) ∧ 
  (∃ m : ℕ, 5^10000 = 1000 * m + 1) ∧
  5^10000 % 1000 = 1 := by sorry

end power_of_five_congruences_and_digits_l3199_319900


namespace butter_production_theorem_l3199_319958

/-- Represents the problem of determining butter production from milk --/
structure MilkButterProblem where
  milk_price : ℚ
  butter_price : ℚ
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  total_earnings : ℚ

/-- Calculates the number of sticks of butter that can be made from one gallon of milk --/
def sticks_per_gallon (p : MilkButterProblem) : ℚ :=
  let total_milk := p.num_cows * p.milk_per_cow
  let sold_milk := p.num_customers * p.milk_per_customer
  let milk_revenue := sold_milk * p.milk_price
  let butter_revenue := p.total_earnings - milk_revenue
  let milk_for_butter := total_milk - sold_milk
  let total_butter_sticks := butter_revenue / p.butter_price
  total_butter_sticks / milk_for_butter

/-- Theorem stating that for the given problem conditions, 2 sticks of butter can be made per gallon of milk --/
theorem butter_production_theorem (p : MilkButterProblem) 
  (h1 : p.milk_price = 3)
  (h2 : p.butter_price = 3/2)
  (h3 : p.num_cows = 12)
  (h4 : p.milk_per_cow = 4)
  (h5 : p.num_customers = 6)
  (h6 : p.milk_per_customer = 6)
  (h7 : p.total_earnings = 144) :
  sticks_per_gallon p = 2 := by
  sorry

end butter_production_theorem_l3199_319958


namespace continuous_functions_integral_bound_l3199_319903

open Set
open MeasureTheory
open Interval

theorem continuous_functions_integral_bound 
  (f g : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g)
  (hf_integral : ∫ x in (Icc 0 1), (f x)^2 = 1)
  (hg_integral : ∫ x in (Icc 0 1), (g x)^2 = 1) :
  ∃ c ∈ Icc 0 1, f c + g c ≤ 2 := by
sorry

end continuous_functions_integral_bound_l3199_319903


namespace modifiedLucas_50th_term_mod_5_l3199_319962

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modifiedLucas_50th_term_mod_5 :
  modifiedLucas 49 % 5 = 0 := by sorry

end modifiedLucas_50th_term_mod_5_l3199_319962


namespace parabola_vertex_l3199_319969

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 + ay + bx + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_vertex :
  let p : Parabola := { equation := fun y x _ => y^2 - 4*y + 3*x + 7 }
  vertex p = (-1, 2) := by sorry

end parabola_vertex_l3199_319969


namespace solution_set_satisfies_inequalities_l3199_319915

def S : Set ℝ := {x | 0 < x ∧ x < 1}

theorem solution_set_satisfies_inequalities :
  ∀ x ∈ S, x * (x + 2) > 0 ∧ |x| < 1 := by sorry

end solution_set_satisfies_inequalities_l3199_319915


namespace variance_equality_and_percentile_l3199_319932

-- Define the sequences x_i and y_i
def x : Fin 10 → ℝ := fun i => 2 * (i.val + 1)
def y : Fin 10 → ℝ := fun i => x i - 20

-- Define variance function
def variance (s : Fin 10 → ℝ) : ℝ := sorry

-- Define percentile function
def percentile (p : ℝ) (s : Fin 10 → ℝ) : ℝ := sorry

theorem variance_equality_and_percentile :
  (variance x = variance y) ∧ (percentile 0.3 y = -13) := by sorry

end variance_equality_and_percentile_l3199_319932


namespace line_tangent_to_circle_l3199_319989

theorem line_tangent_to_circle (b : ℝ) : 
  (∀ x y : ℝ, x - y + b = 0 → (x^2 + y^2 = 25 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      ((x' - y' + b ≠ 0) ∨ (x'^2 + y'^2 ≠ 25)))) → 
  b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by sorry

end line_tangent_to_circle_l3199_319989


namespace calculate_total_exports_l3199_319902

/-- Calculates the total yearly exports of a country given the percentage of fruit exports,
    the percentage of orange exports within fruit exports, and the revenue from orange exports. -/
theorem calculate_total_exports (fruit_export_percent : ℝ) (orange_export_fraction : ℝ) (orange_export_revenue : ℝ) :
  fruit_export_percent = 0.20 →
  orange_export_fraction = 1 / 6 →
  orange_export_revenue = 4.25 →
  (orange_export_revenue / orange_export_fraction) / fruit_export_percent = 127.5 := by
  sorry

end calculate_total_exports_l3199_319902


namespace complex_equation_solution_l3199_319928

theorem complex_equation_solution (z : ℂ) :
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I := by
  sorry

end complex_equation_solution_l3199_319928


namespace ratio_as_percent_l3199_319933

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 25) (h2 : second_part = 50) :
  ∃ (p : ℚ), abs (p - 100 * (first_part : ℚ) / (first_part + second_part)) < 0.01 ∧ p = 33.33 := by
  sorry

end ratio_as_percent_l3199_319933


namespace final_women_count_room_population_problem_l3199_319948

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Represents the changes in population -/
structure PopulationChange where
  menEntered : ℕ
  womenLeft : ℕ
  womenMultiplier : ℕ

/-- The theorem to prove -/
theorem final_women_count 
  (initialRatio : Rat) 
  (changes : PopulationChange) 
  (finalMenCount : ℕ) : ℕ :=
  by
    sorry

/-- The main theorem that encapsulates the problem -/
theorem room_population_problem : 
  final_women_count (7/8) ⟨4, 5, 3⟩ 16 = 27 :=
  by
    sorry

end final_women_count_room_population_problem_l3199_319948


namespace team_B_is_better_l3199_319960

/-- Represents the expected cost of drug development for Team A -/
def expected_cost_A (p : ℝ) (m : ℝ) : ℝ :=
  -2 * m * p^2 + 6 * m

/-- Represents the expected cost of drug development for Team B -/
def expected_cost_B (q : ℝ) (n : ℝ) : ℝ :=
  6 * n * q^3 - 9 * n * q^2 + 6 * n

/-- Theorem stating that Team B's expected cost is less than Team A's when n = 2/3m and p = q -/
theorem team_B_is_better (p q m n : ℝ) 
  (h1 : 0 < p ∧ p < 1) 
  (h2 : m > 0) 
  (h3 : n = 2/3 * m) 
  (h4 : p = q) : 
  expected_cost_B q n < expected_cost_A p m :=
sorry

end team_B_is_better_l3199_319960


namespace inequality_proof_l3199_319984

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) 
  (h_n_pos : n > 0) : 
  (x^4 / (y*(1-y^n))) + (y^4 / (z*(1-z^n))) + (z^4 / (x*(1-x^n))) ≥ 3^n / (3^(n-2) - 9) := by
  sorry

end inequality_proof_l3199_319984


namespace pyramid_division_volumes_l3199_319922

/-- Right quadrangular pyramid with inscribed prism and dividing plane -/
structure PyramidWithPrism where
  /-- Side length of the pyramid's base -/
  a : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Side length of the prism's base -/
  b : ℝ
  /-- Height of the prism -/
  h₀ : ℝ
  /-- Condition: The side length of the pyramid's base is 8√2 -/
  ha : a = 8 * Real.sqrt 2
  /-- Condition: The height of the pyramid is 4 -/
  hh : h = 4
  /-- Condition: The side length of the prism's base is 2 -/
  hb : b = 2
  /-- Condition: The height of the prism is 1 -/
  hh₀ : h₀ = 1

/-- Theorem stating the volumes of the parts divided by plane γ -/
theorem pyramid_division_volumes (p : PyramidWithPrism) :
  ∃ (v₁ v₂ : ℝ), v₁ = 512 / 15 ∧ v₂ = 2048 / 15 ∧
  v₁ + v₂ = (1 / 3) * p.a^2 * p.h :=
sorry

end pyramid_division_volumes_l3199_319922


namespace penguin_colony_size_l3199_319918

/-- Represents the number of penguins in a colony over time -/
structure PenguinColony where
  initial : ℕ
  current : ℕ

/-- Calculates the current number of penguins based on initial conditions -/
def calculate_current_penguins (initial : ℕ) : ℕ :=
  6 * initial + 129

/-- Theorem stating the current number of penguins in the colony -/
theorem penguin_colony_size (colony : PenguinColony) : 
  colony.initial * 3/2 = 237 → colony.current = 1077 := by
  sorry

#check penguin_colony_size

end penguin_colony_size_l3199_319918


namespace distance_city_A_to_B_l3199_319925

/-- The distance between city A and city B given the travel times and speeds of Eddy and Freddy -/
theorem distance_city_A_to_B 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (distance_AC : ℝ) 
  (speed_ratio : ℝ) : 
  time_eddy = 3 → 
  time_freddy = 4 → 
  distance_AC = 300 → 
  speed_ratio = 2.1333333333333333 → 
  time_eddy * (speed_ratio * (distance_AC / time_freddy)) = 480 :=
by sorry

end distance_city_A_to_B_l3199_319925


namespace weeks_of_feed_l3199_319955

-- Define the given quantities
def boxes_bought : ℕ := 3
def boxes_in_pantry : ℕ := 5
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Calculate total boxes and total grams
def total_boxes : ℕ := boxes_bought + boxes_in_pantry
def total_grams : ℕ := total_boxes * grams_per_box

-- Calculate weekly consumption
def weekly_consumption : ℕ := parrot_consumption + cockatiel_consumption

-- Theorem to prove
theorem weeks_of_feed : (total_grams / weekly_consumption : ℕ) = 12 := by
  sorry

end weeks_of_feed_l3199_319955


namespace divisibility_of_powers_l3199_319946

theorem divisibility_of_powers (a : ℕ) (ha : a > 0) :
  ∃ b : ℕ, b > a ∧ (1 + 2^b + 3^b) % (1 + 2^a + 3^a) = 0 :=
by sorry

end divisibility_of_powers_l3199_319946


namespace real_part_of_Z_l3199_319910

theorem real_part_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.abs (3 + 4 * Complex.I)) : 
  Z.re = 5 / 2 := by
  sorry

end real_part_of_Z_l3199_319910


namespace journey_time_bounds_l3199_319983

/-- Represents the bus journey from Kimovsk to Moscow -/
structure BusJourney where
  speed : ℝ
  kimovsk_novomoskovsk : ℝ
  novomoskovsk_tula : ℝ
  tula_moscow : ℝ
  kimovsk_tula_time : ℝ
  novomoskovsk_moscow_time : ℝ

/-- The conditions of the bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.speed ≤ 60 ∧
  j.kimovsk_novomoskovsk = 35 ∧
  j.novomoskovsk_tula = 60 ∧
  j.tula_moscow = 200 ∧
  j.kimovsk_tula_time = 2 ∧
  j.novomoskovsk_moscow_time = 5

/-- The theorem stating the bounds of the total journey time -/
theorem journey_time_bounds (j : BusJourney) 
  (h : journey_conditions j) : 
  ∃ (t : ℝ), 5 + 7/12 ≤ t ∧ t ≤ 6 ∧ 
  t = (j.kimovsk_novomoskovsk + j.novomoskovsk_tula + j.tula_moscow) / j.speed :=
sorry

end journey_time_bounds_l3199_319983


namespace age_gap_ratio_l3199_319973

/-- Given the birth years of family members, prove the ratio of age gaps -/
theorem age_gap_ratio (older_brother_birth : ℕ) (older_sister_birth : ℕ) (grandmother_birth : ℕ)
  (h1 : older_brother_birth = 1932)
  (h2 : older_sister_birth = 1936)
  (h3 : grandmother_birth = 1944) :
  (grandmother_birth - older_sister_birth) / (older_sister_birth - older_brother_birth) = 2 := by
  sorry

end age_gap_ratio_l3199_319973


namespace system_solutions_l3199_319977

theorem system_solutions (x y z : ℝ) : 
  (x * (3 * y^2 + 1) = y * (y^2 + 3) ∧
   y * (3 * z^2 + 1) = z * (z^2 + 3) ∧
   z * (3 * x^2 + 1) = x * (x^2 + 3)) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
   (x = -1 ∧ y = -1 ∧ z = -1) ∨ 
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end system_solutions_l3199_319977


namespace smallest_nonnegative_a_l3199_319956

theorem smallest_nonnegative_a (b : ℝ) (a : ℝ) (h1 : b = π / 4) 
  (h2 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 0 ∧ a = 17 - π / 4 ∧ ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ a :=
by sorry

end smallest_nonnegative_a_l3199_319956


namespace inverse_sum_equals_negative_eight_l3199_319982

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_sum_equals_negative_eight :
  ∃ (a b : ℝ), f a = 4 ∧ f b = -100 ∧ a + b = -8 := by
  sorry

end inverse_sum_equals_negative_eight_l3199_319982


namespace largest_non_prime_sequence_l3199_319937

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit positive integer -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The theorem stating the largest number in the sequence -/
theorem largest_non_prime_sequence :
  ∃ (n : ℕ), 
    (∀ k : ℕ, k ∈ Finset.range 7 → is_two_digit (n - k)) ∧ 
    (∀ k : ℕ, k ∈ Finset.range 7 → n - k < 50) ∧
    (∀ k : ℕ, k ∈ Finset.range 7 → ¬(is_prime (n - k))) ∧
    n = 30 := by
  sorry

end largest_non_prime_sequence_l3199_319937


namespace inequality_implies_range_l3199_319964

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end inequality_implies_range_l3199_319964


namespace exponent_power_rule_l3199_319931

theorem exponent_power_rule (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end exponent_power_rule_l3199_319931


namespace x_plus_y_values_l3199_319994

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end x_plus_y_values_l3199_319994


namespace evaluate_expression_l3199_319905

theorem evaluate_expression : 8^3 + 4*(8^2) + 6*8 + 3 = 1000 := by sorry

end evaluate_expression_l3199_319905


namespace positive_A_value_l3199_319911

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end positive_A_value_l3199_319911


namespace negation_of_all_children_good_l3199_319926

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (GoodAtMusic : U → Prop)

-- Define the original statement and its negation
def AllChildrenGood : Prop := ∀ x, Child x → GoodAtMusic x
def AllChildrenPoor : Prop := ∀ x, Child x → ¬GoodAtMusic x

-- Theorem statement
theorem negation_of_all_children_good :
  AllChildrenPoor U Child GoodAtMusic ↔ ¬AllChildrenGood U Child GoodAtMusic :=
sorry

end negation_of_all_children_good_l3199_319926


namespace contrapositive_equivalence_l3199_319963

theorem contrapositive_equivalence (f : ℝ → ℝ) (a b : ℝ) :
  (¬(f a + f b ≥ f (-a) + f (-b)) → ¬(a + b ≥ 0)) ↔
  (f a + f b < f (-a) + f (-b) → a + b < 0) :=
sorry

end contrapositive_equivalence_l3199_319963


namespace negation_of_proposition_l3199_319995

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔ 
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l3199_319995


namespace last_digit_of_fraction_l3199_319968

def last_digit (n : ℚ) : ℕ := sorry

theorem last_digit_of_fraction :
  last_digit (1 / (2^10 * 3^10)) = 5 := by sorry

end last_digit_of_fraction_l3199_319968


namespace triangle_properties_l3199_319947

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation holds for the triangle -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) / (t.a^2 + t.b^2 - t.c^2) = t.c / (Real.sqrt 2 * t.a - t.c)

theorem triangle_properties (t : Triangle) (h : satisfies_equation t) :
  t.B = π/4 ∧ 
  (t.b = 1 → ∃ (max_area : ℝ), max_area = (Real.sqrt 2 + 1) / 4 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.c * Real.sin t.B → area ≤ max_area) :=
sorry

end triangle_properties_l3199_319947


namespace tammy_second_day_speed_l3199_319972

/-- Represents Tammy's mountain climbing over two days -/
structure MountainClimb where
  total_time : ℝ
  speed_increase : ℝ
  time_decrease : ℝ
  uphill_speed_decrease : ℝ
  downhill_speed_increase : ℝ
  total_distance : ℝ

/-- Calculates Tammy's average speed on the second day -/
def second_day_speed (climb : MountainClimb) : ℝ :=
  -- Definition to be proved
  4

/-- Theorem stating that Tammy's average speed on the second day was 4 km/h -/
theorem tammy_second_day_speed (climb : MountainClimb) 
  (h1 : climb.total_time = 14)
  (h2 : climb.speed_increase = 0.5)
  (h3 : climb.time_decrease = 2)
  (h4 : climb.uphill_speed_decrease = 1)
  (h5 : climb.downhill_speed_increase = 1)
  (h6 : climb.total_distance = 52) :
  second_day_speed climb = 4 := by
  sorry

end tammy_second_day_speed_l3199_319972


namespace zoo_field_trip_remaining_individuals_l3199_319965

/-- Represents the number of individuals from a school -/
structure SchoolGroup :=
  (students : ℕ)
  (parents : ℕ)
  (teachers : ℕ)

/-- Calculates the total number of individuals in a school group -/
def SchoolGroup.total (sg : SchoolGroup) : ℕ :=
  sg.students + sg.parents + sg.teachers

theorem zoo_field_trip_remaining_individuals
  (school_a : SchoolGroup)
  (school_b : SchoolGroup)
  (school_c : SchoolGroup)
  (school_d : SchoolGroup)
  (h1 : school_a = ⟨10, 5, 2⟩)
  (h2 : school_b = ⟨12, 3, 2⟩)
  (h3 : school_c = ⟨15, 3, 0⟩)
  (h4 : school_d = ⟨20, 4, 0⟩)
  (left_students_ab : ℕ)
  (left_students_c : ℕ)
  (left_students_d : ℕ)
  (left_parents_a : ℕ)
  (left_parents_c : ℕ)
  (h5 : left_students_ab = 10)
  (h6 : left_students_c = 6)
  (h7 : left_students_d = 9)
  (h8 : left_parents_a = 2)
  (h9 : left_parents_c = 1)
  : (school_a.total + school_b.total + school_c.total + school_d.total) -
    (left_students_ab + left_students_c + left_students_d + left_parents_a + left_parents_c) = 48 :=
by
  sorry


end zoo_field_trip_remaining_individuals_l3199_319965


namespace symmetric_point_coordinates_l3199_319998

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if two points are symmetric with respect to the origin -/
def is_symmetric_to_origin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Theorem stating that if a point is in the fourth quadrant and symmetric to the origin,
    its symmetric point has negative x and positive y coordinates -/
theorem symmetric_point_coordinates (p : Point) :
  is_in_fourth_quadrant p → ∃ q : Point, is_symmetric_to_origin p q ∧ q.x < 0 ∧ q.y > 0 := by
  sorry


end symmetric_point_coordinates_l3199_319998


namespace hyperbola_properties_l3199_319986

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - 1)^2 / a^2) - ((y - 1)^2 / b^2) = 1

-- Define the conditions
theorem hyperbola_properties :
  ∃ (t : ℝ),
    -- Center at (1, 1) is implicit in the hyperbola definition
    hyperbola 4 2 ∧  -- Passes through (4, 2)
    hyperbola 3 1 ∧  -- Vertex at (3, 1)
    hyperbola t 4 ∧  -- Passes through (t, 4)
    (t^2 = 64 ∨ t^2 = 36) :=
by sorry


end hyperbola_properties_l3199_319986


namespace zongzi_prices_l3199_319942

-- Define variables
variable (x : ℝ) -- Purchase price of egg yolk zongzi
variable (y : ℝ) -- Purchase price of red bean zongzi
variable (m : ℝ) -- Selling price of egg yolk zongzi

-- Define conditions
def first_purchase : Prop := 60 * x + 90 * y = 4800
def second_purchase : Prop := 40 * x + 80 * y = 3600
def initial_sales : Prop := m = 70 ∧ (70 - 50) * 20 = 400
def sales_change : Prop := ∀ p, (p - 50) * (20 + 5 * (70 - p)) = 220 → p = 52

-- Theorem statement
theorem zongzi_prices :
  first_purchase x y ∧ second_purchase x y ∧ initial_sales m ∧ sales_change →
  x = 50 ∧ y = 20 ∧ m = 52 := by
  sorry

end zongzi_prices_l3199_319942


namespace players_sold_is_two_l3199_319951

/-- Represents the financial transactions of a football club --/
def football_club_transactions 
  (initial_balance : ℚ) 
  (selling_price : ℚ) 
  (buying_price : ℚ) 
  (players_bought : ℕ) 
  (final_balance : ℚ) : Prop :=
  ∃ (players_sold : ℕ), 
    initial_balance + (selling_price * players_sold) - (buying_price * players_bought) = final_balance

/-- Theorem stating that the number of players sold is 2 --/
theorem players_sold_is_two : 
  football_club_transactions 100 10 15 4 60 → 
  ∃ (players_sold : ℕ), players_sold = 2 := by
  sorry

end players_sold_is_two_l3199_319951
