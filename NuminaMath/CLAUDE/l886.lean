import Mathlib

namespace truck_speed_through_tunnel_l886_88618

/-- Calculates the speed of a truck passing through a tunnel -/
theorem truck_speed_through_tunnel 
  (truck_length : ℝ) 
  (tunnel_length : ℝ) 
  (exit_time : ℝ) 
  (feet_per_mile : ℝ) 
  (h1 : truck_length = 66) 
  (h2 : tunnel_length = 330) 
  (h3 : exit_time = 6) 
  (h4 : feet_per_mile = 5280) :
  (truck_length + tunnel_length) / exit_time * 3600 / feet_per_mile = 45 := by
sorry

end truck_speed_through_tunnel_l886_88618


namespace probability_one_defective_six_two_two_l886_88699

/-- The probability of selecting exactly one defective product from a set of items. -/
def probability_one_defective (total_items defective_items items_selected : ℕ) : ℚ :=
  let favorable_outcomes := (total_items - defective_items).choose (items_selected - 1) * defective_items.choose 1
  let total_outcomes := total_items.choose items_selected
  favorable_outcomes / total_outcomes

/-- Theorem: The probability of selecting exactly one defective product when taking 2 items at random from 6 items with 2 defective products is 8/15. -/
theorem probability_one_defective_six_two_two :
  probability_one_defective 6 2 2 = 8 / 15 := by
  sorry

end probability_one_defective_six_two_two_l886_88699


namespace log_comparison_l886_88687

theorem log_comparison : 
  (Real.log 4 / Real.log 3) > (0.3 ^ 4) ∧ (0.3 ^ 4) > (Real.log 0.9 / Real.log 1.1) := by
sorry

end log_comparison_l886_88687


namespace homologous_pair_from_both_parents_l886_88642

/-- Represents a parent (mother or father) -/
inductive Parent : Type
| mother : Parent
| father : Parent

/-- Represents a chromosome -/
structure Chromosome : Type :=
  (source : Parent)

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair : Type :=
  (chromosome1 : Chromosome)
  (chromosome2 : Chromosome)

/-- Represents a diploid cell -/
structure DiploidCell : Type :=
  (chromosomePairs : List HomologousPair)

/-- Theorem: In a diploid organism, each pair of homologous chromosomes
    is contributed jointly by the two parents -/
theorem homologous_pair_from_both_parents (cell : DiploidCell) :
  ∀ pair ∈ cell.chromosomePairs,
    (pair.chromosome1.source = Parent.mother ∧ pair.chromosome2.source = Parent.father) ∨
    (pair.chromosome1.source = Parent.father ∧ pair.chromosome2.source = Parent.mother) :=
sorry

end homologous_pair_from_both_parents_l886_88642


namespace escalator_rate_calculation_l886_88674

/-- The rate at which the escalator moves, in feet per second -/
def escalator_rate : ℝ := 11

/-- The length of the escalator, in feet -/
def escalator_length : ℝ := 140

/-- The rate at which the person walks, in feet per second -/
def person_walking_rate : ℝ := 3

/-- The time taken by the person to cover the entire length, in seconds -/
def time_taken : ℝ := 10

theorem escalator_rate_calculation :
  (person_walking_rate + escalator_rate) * time_taken = escalator_length :=
by sorry

end escalator_rate_calculation_l886_88674


namespace prime_square_minus_one_div_24_l886_88609

theorem prime_square_minus_one_div_24 (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end prime_square_minus_one_div_24_l886_88609


namespace sticker_count_l886_88621

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers (ryan_stickers : ℕ) (steven_multiplier : ℕ) (terry_extra : ℕ) : ℕ :=
  ryan_stickers + 
  (steven_multiplier * ryan_stickers) + 
  (steven_multiplier * ryan_stickers + terry_extra)

/-- Proof that the total number of stickers is 230 -/
theorem sticker_count : total_stickers 30 3 20 = 230 := by
  sorry

end sticker_count_l886_88621


namespace daily_shoppers_l886_88627

theorem daily_shoppers (tax_free_percentage : ℝ) (weekly_tax_payers : ℕ) : 
  tax_free_percentage = 0.06 →
  weekly_tax_payers = 6580 →
  ∃ (daily_shoppers : ℕ), daily_shoppers = 1000 ∧ 
    (1 - tax_free_percentage) * (daily_shoppers : ℝ) * 7 = weekly_tax_payers := by
  sorry

end daily_shoppers_l886_88627


namespace min_value_of_x_l886_88675

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 + (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end min_value_of_x_l886_88675


namespace x_power_minus_reciprocal_l886_88637

theorem x_power_minus_reciprocal (θ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x - 1/x = 2 * Real.sin θ) (h4 : n > 0) : 
  x^n - 1/(x^n) = 2 * Real.sinh (n * θ) := by
  sorry

end x_power_minus_reciprocal_l886_88637


namespace function_zero_in_interval_l886_88612

theorem function_zero_in_interval (a : ℝ) : 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ a^2 * x - 2*a + 1 = 0) ↔ 
  a ∈ (Set.Ioo (1/2) 1) ∪ (Set.Ioi 1) := by
sorry

end function_zero_in_interval_l886_88612


namespace exponent_fraction_simplification_l886_88692

theorem exponent_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end exponent_fraction_simplification_l886_88692


namespace cars_distance_theorem_l886_88688

/-- The distance between two cars after their movements on a main road -/
def final_distance (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem stating the final distance between two cars -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  final_distance initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval final_distance 113 50 35

end cars_distance_theorem_l886_88688


namespace people_eating_both_veg_nonveg_l886_88695

/-- The number of people who eat only vegetarian food -/
def only_veg : ℕ := 13

/-- The total number of people who eat vegetarian food -/
def total_veg : ℕ := 21

/-- The number of people who eat both vegetarian and non-vegetarian food -/
def both_veg_nonveg : ℕ := total_veg - only_veg

theorem people_eating_both_veg_nonveg : both_veg_nonveg = 8 := by
  sorry

end people_eating_both_veg_nonveg_l886_88695


namespace negative_seven_times_sum_l886_88678

theorem negative_seven_times_sum : -7 * 45 + (-7) * 55 = -700 := by
  sorry

end negative_seven_times_sum_l886_88678


namespace eggs_per_basket_l886_88640

theorem eggs_per_basket (purple_eggs blue_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) : 
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    purple_eggs % eggs_per_basket = 0 ∧
    blue_eggs % eggs_per_basket = 0 ∧
    eggs_per_basket = 6 := by
  sorry

end eggs_per_basket_l886_88640


namespace similar_triangles_ratio_equality_l886_88611

/-- Two triangles are similar if there exists a complex number k that maps one triangle to the other -/
def similar_triangles (a b c a' b' c' : ℂ) : Prop :=
  ∃ k : ℂ, k ≠ 0 ∧ b - a = k * (b' - a') ∧ c - a = k * (c' - a')

/-- Theorem: For similar triangles abc and a'b'c' on the complex plane, 
    the ratio (b-a)/(c-a) equals (b'-a')/(c'-a') -/
theorem similar_triangles_ratio_equality 
  (a b c a' b' c' : ℂ) 
  (h : similar_triangles a b c a' b' c') 
  (h1 : c ≠ a) 
  (h2 : c' ≠ a') : 
  (b - a) / (c - a) = (b' - a') / (c' - a') := by
  sorry

end similar_triangles_ratio_equality_l886_88611


namespace extreme_values_of_f_l886_88697

def f (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -1) ∧ (∀ x : ℝ, f x ≤ 3) :=
sorry

end extreme_values_of_f_l886_88697


namespace rachel_total_score_l886_88648

/-- Rachel's video game scoring system -/
def video_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_total_score :
  video_game_score 9 5 2 = 63 :=
by sorry

end rachel_total_score_l886_88648


namespace charity_distribution_l886_88629

theorem charity_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) : 
  total_amount = 2500 →
  donation_percentage = 0.80 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end charity_distribution_l886_88629


namespace compute_expression_l886_88647

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end compute_expression_l886_88647


namespace jay_painting_time_l886_88677

theorem jay_painting_time (bong_time : ℝ) (combined_time : ℝ) (jay_time : ℝ) : 
  bong_time = 3 → 
  combined_time = 1.2 → 
  (1 / jay_time) + (1 / bong_time) = (1 / combined_time) → 
  jay_time = 2 := by
sorry

end jay_painting_time_l886_88677


namespace five_twelve_thirteen_right_triangle_l886_88638

/-- A triple of positive integers representing the sides of a triangle -/
structure TripleSides where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triple of sides satisfies the Pythagorean theorem -/
def is_right_triangle (sides : TripleSides) : Prop :=
  (sides.a.val ^ 2 : ℕ) + (sides.b.val ^ 2 : ℕ) = (sides.c.val ^ 2 : ℕ)

/-- The triple (5, 12, 13) forms a right triangle -/
theorem five_twelve_thirteen_right_triangle :
  is_right_triangle ⟨5, 12, 13⟩ := by sorry

end five_twelve_thirteen_right_triangle_l886_88638


namespace function_monotonicity_l886_88668

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2) * x + 2

theorem function_monotonicity (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  4 ≤ a ∧ a < 8 :=
by sorry

end function_monotonicity_l886_88668


namespace inverse_proportion_k_value_l886_88685

theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → k = -6 := by
  sorry

end inverse_proportion_k_value_l886_88685


namespace work_completion_time_l886_88614

/-- Given:
  * A can do a work in 20 days
  * A works for 10 days and then leaves
  * B can finish the remaining work in 15 days
Prove that B can do the entire work in 30 days -/
theorem work_completion_time (a_time b_remaining_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_remaining_time = 15) :
  let a_work_rate : ℚ := 1 / a_time
  let a_work_done : ℚ := a_work_rate * 10
  let remaining_work : ℚ := 1 - a_work_done
  let b_rate : ℚ := remaining_work / b_remaining_time
  b_rate⁻¹ = 30 := by
  sorry

#check work_completion_time

end work_completion_time_l886_88614


namespace f_max_min_l886_88607

-- Define the function
def f (x : ℝ) : ℝ := |-(x)| - |x - 3|

-- State the theorem
theorem f_max_min :
  (∀ x : ℝ, f x ≤ 3) ∧
  (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -3) ∧
  (∃ x : ℝ, f x = -3) :=
sorry

end f_max_min_l886_88607


namespace curve_is_circle_and_line_l886_88625

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ - 3 * ρ * Real.cos θ + ρ - 3 = 0

/-- Definition of a circle in polar coordinates -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

/-- Definition of a line in polar coordinates -/
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ ρ * (a * Real.cos θ + b * Real.sin θ) = 1

/-- The theorem stating that the curve consists of a circle and a line -/
theorem curve_is_circle_and_line :
  (∃ f g : ℝ → ℝ → Prop, 
    (∀ ρ θ : ℝ, polar_equation ρ θ ↔ (f ρ θ ∨ g ρ θ)) ∧
    is_circle f ∧ is_line g) :=
sorry

end curve_is_circle_and_line_l886_88625


namespace tangent_line_to_circle_l886_88689

/-- Given a circle and a line, find the equations of lines tangent to the circle and parallel to the given line -/
theorem tangent_line_to_circle (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 + 2*y = 0}
  let l2 := {(x, y) | 3*x + 4*y - 6 = 0}
  ∃ (b : ℝ), (b = -1 ∨ b = 9) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ {(x, y) | 3*x + 4*y + b = 0} → 
      (∃ (x0 y0 : ℝ), (x0, y0) ∈ circle ∧ 
        ((x - x0)^2 + (y - y0)^2 = 1 ∧
         ∀ (x1 y1 : ℝ), (x1, y1) ∈ circle → (x1 - x0)^2 + (y1 - y0)^2 ≤ 1)))
  := by sorry

end tangent_line_to_circle_l886_88689


namespace unique_solution_system_l886_88628

theorem unique_solution_system (x y z : ℝ) : 
  (Real.sqrt (x - 997) + Real.sqrt (y - 932) + Real.sqrt (z - 796) = 100) ∧
  (Real.sqrt (x - 1237) + Real.sqrt (y - 1121) + Real.sqrt (3045 - z) = 90) ∧
  (Real.sqrt (x - 1621) + Real.sqrt (2805 - y) + Real.sqrt (z - 997) = 80) ∧
  (Real.sqrt (2102 - x) + Real.sqrt (y - 1237) + Real.sqrt (z - 932) = 70) →
  x = 2021 ∧ y = 2021 ∧ z = 2021 :=
by sorry

end unique_solution_system_l886_88628


namespace line_segment_sum_l886_88620

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -3/4 * x + 9

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 9)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (r, s) = (1 - t) • P + t • Q

/-- The area of triangle POQ is three times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 3 * abs ((P.1 * s - r * P.2) / 2)

/-- The main theorem -/
theorem line_segment_sum (r s : ℝ) :
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 11 := by sorry

end line_segment_sum_l886_88620


namespace constant_value_proof_l886_88698

/-- Given consecutive integers x, y, and z where x > y > z, z = 2, 
    and 2x + 3y + 3z = 5y + C, prove that C = 8 -/
theorem constant_value_proof (x y z : ℤ) (C : ℤ) 
    (h1 : x = z + 2)
    (h2 : y = z + 1)
    (h3 : x > y ∧ y > z)
    (h4 : z = 2)
    (h5 : 2*x + 3*y + 3*z = 5*y + C) : C = 8 := by
  sorry

end constant_value_proof_l886_88698


namespace perpendicular_tangents_a_value_l886_88608

/-- The value of 'a' for which the curves y = ax³ - 6x² + 12x and y = exp(x)
    have perpendicular tangents at x = 1 -/
theorem perpendicular_tangents_a_value :
  ∀ a : ℝ,
  (∀ x : ℝ, deriv (fun x => a * x^3 - 6 * x^2 + 12 * x) 1 *
             deriv (fun x => Real.exp x) 1 = -1) →
  a = -1 / (3 * Real.exp 1) := by
sorry


end perpendicular_tangents_a_value_l886_88608


namespace least_area_rectangle_l886_88643

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The least possible area of a rectangle with perimeter 200 and area divisible by 10 is 900 -/
theorem least_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 200 ∧
    area r % 10 = 0 ∧
    area r = 900 ∧
    ∀ (s : Rectangle),
      perimeter s = 200 →
      area s % 10 = 0 →
      area r ≤ area s :=
sorry

end least_area_rectangle_l886_88643


namespace negation_of_proposition_l886_88657

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1) ↔
  (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) := by
  sorry

end negation_of_proposition_l886_88657


namespace min_value_2a6_plus_a5_l886_88632

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = q * a n

/-- The theorem stating the minimum value of 2a_6 + a_5 for a specific geometric sequence -/
theorem min_value_2a6_plus_a5 (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  (2 * a 4 + a 3 = 2 * a 2 + a 1 + 8) →
  (∀ x, 2 * a 6 + a 5 ≥ x) →
  x = 32 := by
  sorry

end min_value_2a6_plus_a5_l886_88632


namespace inverse_of_3_mod_179_l886_88615

theorem inverse_of_3_mod_179 : ∃ x : ℕ, x < 179 ∧ (3 * x) % 179 = 1 :=
by
  use 60
  sorry

#eval (3 * 60) % 179  -- Should output 1

end inverse_of_3_mod_179_l886_88615


namespace legs_on_queen_mary_ii_l886_88604

/-- Calculates the total number of legs on a ship with cats and humans. -/
def total_legs (total_heads : ℕ) (num_cats : ℕ) : ℕ :=
  let num_humans := total_heads - num_cats
  let cat_legs := num_cats * 4
  let human_legs := (num_humans - 1) * 2 + 1
  cat_legs + human_legs

/-- Theorem stating that the total number of legs is 45 under given conditions. -/
theorem legs_on_queen_mary_ii :
  total_legs 16 7 = 45 := by
  sorry

end legs_on_queen_mary_ii_l886_88604


namespace f_max_value_l886_88651

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) : deriv f x = (1 / x^2 - 2 * f x) / x

axiom f_initial_value : f 1 = 2

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = Real.exp 3 / 2 :=
sorry

end f_max_value_l886_88651


namespace basketball_game_scores_l886_88693

/-- Represents the scores of a basketball team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    s.q2 = s.q1 * r ∧
    s.q3 = s.q1 * r^2 ∧
    s.q4 = s.q1 * r^3

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧
    s.q2 = s.q1 + d ∧
    s.q3 = s.q1 + 2*d ∧
    s.q4 = s.q1 + 3*d

/-- The main theorem about the basketball game -/
theorem basketball_game_scores 
  (eagles lions : TeamScores)
  (h1 : eagles.q1 = lions.q1)
  (h2 : is_increasing_geometric eagles)
  (h3 : is_increasing_arithmetic lions)
  (h4 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 = 
        lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2)
  (h5 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100) :
  eagles.q1 + eagles.q2 + lions.q1 + lions.q2 = 43 :=
sorry

end basketball_game_scores_l886_88693


namespace bryce_raisins_l886_88679

theorem bryce_raisins : ∃ (b c : ℕ), b = c + 8 ∧ c = b / 3 → b = 12 := by
  sorry

end bryce_raisins_l886_88679


namespace number_of_boys_l886_88619

theorem number_of_boys (total_children happy_children sad_children neutral_children girls happy_boys sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neutral_children →
  ∃ boys, boys = total_children - girls ∧ boys = 18 :=
by
  sorry

end number_of_boys_l886_88619


namespace shaded_squares_formula_l886_88631

/-- Represents a row of squares in the pattern -/
structure Row :=
  (number : ℕ)  -- The row number
  (total : ℕ)   -- Total number of squares in the row
  (unshaded : ℕ) -- Number of unshaded squares
  (shaded : ℕ)   -- Number of shaded squares

/-- The properties of the sequence of rows -/
def ValidSequence (rows : ℕ → Row) : Prop :=
  (rows 1).total = 1 ∧ 
  (rows 1).unshaded = 1 ∧
  (rows 1).shaded = 0 ∧
  (∀ n : ℕ, n > 0 → (rows n).number = n) ∧
  (∀ n : ℕ, n > 1 → (rows n).total = (rows (n-1)).total + 2) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = (rows n).total - (rows n).shaded) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = n)

theorem shaded_squares_formula (rows : ℕ → Row) 
  (h : ValidSequence rows) (n : ℕ) (hn : n > 0) : 
  (rows n).shaded = n - 1 :=
sorry

end shaded_squares_formula_l886_88631


namespace x_value_proof_l886_88633

theorem x_value_proof (x : ℚ) 
  (h1 : 8 * x^2 + 9 * x - 2 = 0) 
  (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1/8 := by
sorry

end x_value_proof_l886_88633


namespace lollipop_distribution_l886_88613

/-- The number of kids in the group -/
def num_kids : ℕ := 42

/-- The initial number of lollipops available -/
def initial_lollipops : ℕ := 650

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The additional lollipops needed -/
def additional_lollipops : ℕ := sum_first_n num_kids - initial_lollipops

theorem lollipop_distribution :
  additional_lollipops = 253 ∧
  ∀ k, k ≤ num_kids → k ≤ sum_first_n num_kids ∧
  sum_first_n num_kids = initial_lollipops + additional_lollipops :=
sorry

end lollipop_distribution_l886_88613


namespace inverse_of_A_l886_88610

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; -1, -1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/3, -7/3; 1/3, 4/3]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l886_88610


namespace derivative_x_ln_x_l886_88667

open Real

/-- The derivative of x * ln(x) is 1 + ln(x) -/
theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => x * log x) x = 1 + log x := by
  sorry

end derivative_x_ln_x_l886_88667


namespace money_split_ratio_l886_88600

theorem money_split_ratio (parker_share richie_share total : ℚ) : 
  parker_share / richie_share = 2 / 3 →
  parker_share = 50 →
  parker_share < richie_share →
  total = parker_share + richie_share →
  total = 125 := by
sorry

end money_split_ratio_l886_88600


namespace students_walking_home_fraction_l886_88639

theorem students_walking_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/6
  let bike_fraction : ℚ := 1/15
  let total_fraction : ℚ := 1
  let other_transport_fraction : ℚ := bus_fraction + auto_fraction + bike_fraction
  let walking_fraction : ℚ := total_fraction - other_transport_fraction
  walking_fraction = 13/30 := by
sorry

end students_walking_home_fraction_l886_88639


namespace rectangle_combination_forms_square_l886_88649

theorem rectangle_combination_forms_square (n : Nat) (h : n = 100) :
  ∃ (square : Set (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ square → x < n ∧ y < n) ∧ 
    (∀ (x y : ℕ), (x, y) ∈ square → (x + 1, y) ∈ square ∨ (x, y + 1) ∈ square) ∧
    (∃ (x y : ℕ), 
      (x, y) ∈ square ∧ 
      (x + 1, y) ∈ square ∧ 
      (x, y + 1) ∈ square ∧ 
      (x + 1, y + 1) ∈ square) :=
by
  sorry


end rectangle_combination_forms_square_l886_88649


namespace hexagon_angle_A_l886_88671

/-- A hexagon is a polygon with 6 sides -/
def Hexagon (A B C D E F : ℝ) : Prop :=
  A + B + C + D + E + F = 720

/-- The theorem states that in a hexagon ABCDEF where B = 134°, C = 98°, D = 120°, E = 139°, and F = 109°, the measure of angle A is 120° -/
theorem hexagon_angle_A (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) : 
  A = 120 := by
  sorry

end hexagon_angle_A_l886_88671


namespace rectangular_field_area_l886_88680

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length > 0 → 
  width = length / 3 → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  width * length = 243 := by
sorry

end rectangular_field_area_l886_88680


namespace hcf_of_numbers_l886_88634

def number1 : ℕ := 210
def number2 : ℕ := 330
def lcm_value : ℕ := 2310

theorem hcf_of_numbers : Nat.gcd number1 number2 = 30 := by
  sorry

end hcf_of_numbers_l886_88634


namespace greatest_cars_with_ac_no_stripes_l886_88659

theorem greatest_cars_with_ac_no_stripes (total : Nat) (no_ac : Nat) (min_stripes : Nat)
  (h1 : total = 100)
  (h2 : no_ac = 37)
  (h3 : min_stripes = 51)
  (h4 : min_stripes ≤ total)
  (h5 : no_ac < total) :
  (total - no_ac) - min_stripes = 12 :=
sorry

end greatest_cars_with_ac_no_stripes_l886_88659


namespace b_age_l886_88644

-- Define variables for ages
variable (a b c : ℕ)

-- Define the conditions from the problem
axiom age_relation : a = b + 2
axiom b_twice_c : b = 2 * c
axiom total_age : a + b + c = 27

-- Theorem to prove
theorem b_age : b = 10 := by
  sorry

end b_age_l886_88644


namespace wizard_elixir_combinations_l886_88623

/-- The number of enchanted herbs available to the wizard. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available to the wizard. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with the incompatible crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 := by
  sorry

end wizard_elixir_combinations_l886_88623


namespace problem_solution_l886_88622

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end problem_solution_l886_88622


namespace equilateral_triangle_rotation_volume_l886_88656

/-- The volume of a solid obtained by rotating an equilateral triangle around a line parallel to its altitude -/
theorem equilateral_triangle_rotation_volume (a : ℝ) (ha : a > 0) :
  let h := a * Real.sqrt 3 / 2
  let r := a / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * a^3 * Real.sqrt 3) / 24 :=
by sorry

end equilateral_triangle_rotation_volume_l886_88656


namespace isha_pencil_length_l886_88654

/-- The length of a pencil after sharpening, given its original length and the length sharpened off. -/
def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

/-- Theorem stating that a 31-inch pencil sharpened by 17 inches results in a 14-inch pencil. -/
theorem isha_pencil_length :
  pencil_length_after_sharpening 31 17 = 14 := by
  sorry

end isha_pencil_length_l886_88654


namespace anna_ate_three_cupcakes_l886_88662

def total_cupcakes : ℕ := 60
def fraction_given_away : ℚ := 4/5
def cupcakes_left : ℕ := 9

theorem anna_ate_three_cupcakes :
  total_cupcakes - (fraction_given_away * total_cupcakes).floor - cupcakes_left = 3 := by
  sorry

end anna_ate_three_cupcakes_l886_88662


namespace exists_perpendicular_plane_containing_line_l886_88652

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a plane contains a line -/
def contains_line (β : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are perpendicular -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line intersects a plane but is not perpendicular to it,
    then there exists a plane containing the line that is perpendicular to the original plane -/
theorem exists_perpendicular_plane_containing_line
  (l : Line3D) (α : Plane3D)
  (h1 : intersects l α)
  (h2 : ¬perpendicular_line_plane l α) :
  ∃ β : Plane3D, contains_line β l ∧ perpendicular_planes α β :=
sorry

end exists_perpendicular_plane_containing_line_l886_88652


namespace f_negative_two_l886_88664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + x^3 + 1

theorem f_negative_two (a : ℝ) (h : f a 2 = 3) : f a (-2) = -1 := by
  sorry

end f_negative_two_l886_88664


namespace expression_value_l886_88690

def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end expression_value_l886_88690


namespace translate_sin_function_l886_88650

/-- Translates the given trigonometric function and proves the result -/
theorem translate_sin_function :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6) + 1
  ∀ x, g x = 2 * (Real.cos x) ^ 2 := by
  sorry

end translate_sin_function_l886_88650


namespace zero_in_interval_l886_88626

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 - 8 + 2 * x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 3 4, f c = 0 :=
by
  sorry

end zero_in_interval_l886_88626


namespace volume_to_surface_area_ratio_l886_88676

/-- Represents a configuration of unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  num_outlying : ℕ
  num_exposed_faces : ℕ

/-- Calculates the volume of a cube configuration -/
def volume (config : CubeConfiguration) : ℕ := config.num_cubes

/-- Calculates the surface area of a cube configuration -/
def surface_area (config : CubeConfiguration) : ℕ := config.num_exposed_faces

/-- The specific configuration described in the problem -/
def problem_config : CubeConfiguration :=
  { num_cubes := 8
  , num_outlying := 7
  , num_exposed_faces := 33 }

/-- Theorem stating the ratio of volume to surface area for the given configuration -/
theorem volume_to_surface_area_ratio (config : CubeConfiguration) :
  config = problem_config →
  (volume config : ℚ) / (surface_area config : ℚ) = 8 / 33 := by
  sorry

end volume_to_surface_area_ratio_l886_88676


namespace harry_anna_pencil_ratio_l886_88658

/-- Proves that the ratio of Harry's initial pencils to Anna's pencils is 2:1 --/
theorem harry_anna_pencil_ratio :
  ∀ (anna_pencils : ℕ) (harry_initial : ℕ) (harry_lost : ℕ) (harry_left : ℕ),
    anna_pencils = 50 →
    harry_initial = anna_pencils * harry_left / (anna_pencils - harry_lost) →
    harry_lost = 19 →
    harry_left = 81 →
    harry_initial / anna_pencils = 2 := by
  sorry

end harry_anna_pencil_ratio_l886_88658


namespace theater_ticket_cost_is_3320_l886_88683

/-- Calculates the total cost of theater tickets sold given the following conditions:
    - Total tickets sold: 370
    - Orchestra ticket price: $12
    - Balcony ticket price: $8
    - 190 more balcony tickets sold than orchestra tickets
-/
def theater_ticket_cost : ℕ := by
  -- Define the total number of tickets sold
  let total_tickets : ℕ := 370
  -- Define the price of orchestra tickets
  let orchestra_price : ℕ := 12
  -- Define the price of balcony tickets
  let balcony_price : ℕ := 8
  -- Define the difference between balcony and orchestra tickets sold
  let balcony_orchestra_diff : ℕ := 190
  
  -- Calculate the number of orchestra tickets sold
  let orchestra_tickets : ℕ := (total_tickets - balcony_orchestra_diff) / 2
  -- Calculate the number of balcony tickets sold
  let balcony_tickets : ℕ := total_tickets - orchestra_tickets
  
  -- Calculate and return the total cost
  exact orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

/-- Theorem stating that the total cost of theater tickets is $3320 -/
theorem theater_ticket_cost_is_3320 : theater_ticket_cost = 3320 := by
  sorry

end theater_ticket_cost_is_3320_l886_88683


namespace absolute_value_equation_unique_solution_l886_88682

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
sorry

end absolute_value_equation_unique_solution_l886_88682


namespace sum_of_three_numbers_l886_88661

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end sum_of_three_numbers_l886_88661


namespace rational_roots_of_polynomial_l886_88696

theorem rational_roots_of_polynomial (x : ℚ) :
  (3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0) ↔ (x = 1 ∨ x = 1/3) :=
by sorry

end rational_roots_of_polynomial_l886_88696


namespace collinear_points_sum_l886_88694

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop := sorry

theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end collinear_points_sum_l886_88694


namespace combined_shape_perimeter_l886_88669

/-- Given a shape consisting of a rectangle and a right triangle sharing one side,
    where the rectangle has sides of length 6 and x, and the triangle has legs of length x and 6,
    the perimeter of the combined shape is 18 + 2x + √(x^2 + 36). -/
theorem combined_shape_perimeter (x : ℝ) :
  let rectangle_perimeter := 2 * (6 + x)
  let triangle_hypotenuse := Real.sqrt (x^2 + 36)
  let shared_side := x
  rectangle_perimeter + x + 6 + triangle_hypotenuse - shared_side = 18 + 2*x + Real.sqrt (x^2 + 36) := by
  sorry

end combined_shape_perimeter_l886_88669


namespace inscribed_circle_radius_l886_88645

-- Define the right triangle XYZ
def XYZ : Set (ℝ × ℝ) := sorry

-- Define the lengths of the sides
def XZ : ℝ := 15
def YZ : ℝ := 8

-- Define that Z is a right angle
def Z_is_right_angle : sorry := sorry

-- Define the inscribed circle
def inscribed_circle : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem inscribed_circle_radius :
  ∃ (r : ℝ), r = 3 ∧ ∀ (p : ℝ × ℝ), p ∈ inscribed_circle → 
    ∃ (c : ℝ × ℝ), c ∈ XYZ ∧ dist p c = r :=
sorry

end inscribed_circle_radius_l886_88645


namespace max_value_of_expression_max_value_achievable_l886_88603

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) ≤ Real.sqrt 28 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) = Real.sqrt 28 :=
by sorry

end max_value_of_expression_max_value_achievable_l886_88603


namespace arithmetic_sequence_sum_l886_88684

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 7 + a 13 = 4 →
  a 2 + a 12 = 8/3 := by
sorry

end arithmetic_sequence_sum_l886_88684


namespace parabola_intersection_l886_88601

/-- Two parabolas with different vertices have equations y = px^2 and y = q(x-a)^2 + b, 
    where (0,0) is the vertex of the first parabola and (a,b) is the vertex of the second parabola. 
    Each vertex lies on the other parabola. -/
theorem parabola_intersection (p q a b : ℝ) (h1 : a ≠ 0) (h2 : b = p * a^2) (h3 : 0 = q * a^2 + b) : 
  p + q = 0 := by sorry

end parabola_intersection_l886_88601


namespace complex_fraction_simplification_l886_88653

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 3 * i) / (4 - 5 * i) = 23 / 41 - (2 / 41) * i := by
  sorry

end complex_fraction_simplification_l886_88653


namespace bargain_bin_books_l886_88616

theorem bargain_bin_books (initial_books sold_books added_books remaining_books : ℕ) :
  initial_books - sold_books + added_books = remaining_books →
  sold_books = 33 →
  added_books = 2 →
  remaining_books = 10 →
  initial_books = 41 :=
by
  sorry

end bargain_bin_books_l886_88616


namespace min_value_theorem_l886_88681

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 2 / y) ≥ 4 := by
  sorry

end min_value_theorem_l886_88681


namespace negation_of_universal_proposition_l886_88641

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2 → x^2 ≥ 4) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) := by
  sorry

end negation_of_universal_proposition_l886_88641


namespace system_solution_l886_88605

theorem system_solution (x y m : ℚ) : 
  x + 3 * y = 7 ∧ 
  x - 3 * y + m * x + 3 = 0 ∧ 
  2 * x - 3 * y = 2 → 
  m = -2/3 := by sorry

end system_solution_l886_88605


namespace right_triangle_acute_angles_l886_88672

theorem right_triangle_acute_angles (α β : ℝ) : 
  α + β = 90 →  -- sum of acute angles in a right triangle is 90°
  α = 54 →      -- one acute angle is 54°
  β = 36 :=     -- the other acute angle is 36°
by sorry

end right_triangle_acute_angles_l886_88672


namespace x_intercept_of_line_l886_88617

/-- The x-intercept of a line passing through two given points is -3/2 -/
theorem x_intercept_of_line (p1 p2 : ℝ × ℝ) : 
  p1 = (-1, 1) → p2 = (0, 3) → 
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ (x, y) ∈ ({p1, p2} : Set (ℝ × ℝ))) → 
  (0 = m * (-3/2) + b) := by
  sorry

end x_intercept_of_line_l886_88617


namespace boundaries_hit_count_l886_88691

/-- Represents the number of runs scored by a boundary -/
def boundary_runs : ℕ := 4

/-- Represents the number of runs scored by a six -/
def six_runs : ℕ := 6

/-- Represents the total runs scored by the batsman -/
def total_runs : ℕ := 120

/-- Represents the number of sixes hit by the batsman -/
def sixes_hit : ℕ := 8

/-- Represents the fraction of runs scored by running between wickets -/
def running_fraction : ℚ := 1/2

theorem boundaries_hit_count :
  ∃ (boundaries : ℕ),
    boundaries * boundary_runs + 
    sixes_hit * six_runs + 
    (running_fraction * total_runs).num = total_runs ∧
    boundaries = 3 := by sorry

end boundaries_hit_count_l886_88691


namespace inequality_proof_l886_88686

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 + 1/x) * (1 + 1/y) ≥ 4 := by
  sorry

end inequality_proof_l886_88686


namespace sisters_ages_l886_88666

theorem sisters_ages (a b c : ℕ+) 
  (middle_age : c = 10)
  (age_relation : a * 10 - 9 * b = 89) :
  a = 17 ∧ b = 9 ∧ c = 10 := by
sorry

end sisters_ages_l886_88666


namespace erdos_szekeres_101_l886_88673

theorem erdos_szekeres_101 (σ : Fin 101 → Fin 101) :
  ∃ (s : Finset (Fin 101)) (f : Fin 11 → Fin 101),
    s.card = 11 ∧ 
    (∀ i j : Fin 11, i < j → (f i : ℕ) < (f j : ℕ) ∨ (f i : ℕ) > (f j : ℕ)) ∧
    (∀ i : Fin 11, f i ∈ s) :=
sorry

end erdos_szekeres_101_l886_88673


namespace cube_root_problem_l886_88635

theorem cube_root_problem (a : ℝ) (h : a^3 = 21 * 25 * 15 * 147) : a = 105 := by
  sorry

end cube_root_problem_l886_88635


namespace sum_f_negative_l886_88630

def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end sum_f_negative_l886_88630


namespace binary_1010101_equals_85_l886_88602

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010 101₍₂₎ -/
def binary_number : List Bool := [true, false, true, false, true, false, true]

/-- Theorem stating that 1010 101₍₂₎ is equal to 85 in decimal -/
theorem binary_1010101_equals_85 : binary_to_decimal binary_number = 85 := by
  sorry

end binary_1010101_equals_85_l886_88602


namespace workday_meeting_percentage_l886_88665

/-- Calculates the percentage of a workday spent in meetings given the workday length and meeting durations. -/
theorem workday_meeting_percentage 
  (workday_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_multiplier : ℕ) : 
  workday_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 3 →
  (first_meeting_minutes + first_meeting_minutes * second_meeting_multiplier) / (workday_hours * 60) * 100 = 40 := by
  sorry

end workday_meeting_percentage_l886_88665


namespace sum_convergence_l886_88670

/-- Given two sequences of real numbers (a_n) and (b_n) satisfying the condition
    (3 - 2i)^n = a_n + b_ni for all integers n ≥ 0, where i = √(-1),
    prove that the sum ∑(n=0 to ∞) (a_n * b_n) / 8^n converges to 4/5. -/
theorem sum_convergence (a b : ℕ → ℝ) 
    (h : ∀ n : ℕ, Complex.I ^ 2 = -1 ∧ (3 - 2 * Complex.I) ^ n = a n + b n * Complex.I) :
    HasSum (λ n => (a n * b n) / 8^n) (4/5) :=
by sorry

end sum_convergence_l886_88670


namespace coin_flip_probability_l886_88660

theorem coin_flip_probability (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 5/32 ↔ n = 6 :=
by sorry

end coin_flip_probability_l886_88660


namespace eighth_odd_multiple_of_5_l886_88624

/-- A function that generates the nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- The property of being both odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem eighth_odd_multiple_of_5 :
  nthOddMultipleOf5 8 = 75 ∧ 
  isOddMultipleOf5 (nthOddMultipleOf5 8) ∧
  (∀ m < 8, ∃ k < nthOddMultipleOf5 8, k > 0 ∧ isOddMultipleOf5 k) :=
sorry

end eighth_odd_multiple_of_5_l886_88624


namespace gain_percent_calculation_l886_88646

/-- If the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is (3/7) * 100. -/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 35 * S) :
  (S - C) / C * 100 = (3 / 7) * 100 := by
  sorry

end gain_percent_calculation_l886_88646


namespace max_sundays_in_fifty_days_l886_88655

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of days we're considering -/
def daysConsidered : ℕ := 50

/-- The maximum number of Sundays in the first 50 days of any year -/
def maxSundays : ℕ := daysConsidered / daysInWeek

theorem max_sundays_in_fifty_days :
  maxSundays = 7 := by sorry

end max_sundays_in_fifty_days_l886_88655


namespace intersection_M_N_l886_88663

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end intersection_M_N_l886_88663


namespace parallel_vectors_imply_x_half_l886_88636

/-- Two vectors are parallel if their cross product is zero -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_imply_x_half :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, x)
  IsParallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) →
  x = 1/2 := by
sorry

end parallel_vectors_imply_x_half_l886_88636


namespace g_at_minus_one_l886_88606

/-- The function g(x) = -2x^2 + 5x - 7 --/
def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

/-- Theorem: g(-1) = -14 --/
theorem g_at_minus_one : g (-1) = -14 := by
  sorry

end g_at_minus_one_l886_88606
