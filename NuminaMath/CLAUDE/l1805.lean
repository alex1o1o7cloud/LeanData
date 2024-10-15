import Mathlib

namespace NUMINAMATH_CALUDE_smallest_power_congruence_l1805_180560

theorem smallest_power_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 100 → (2013 ^ m) % 1000 ≠ 1) ∧ 
  (2013 ^ 100) % 1000 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l1805_180560


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1805_180548

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x : ℝ | 3 < x ∧ x < 9}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1805_180548


namespace NUMINAMATH_CALUDE_friends_with_boxes_eq_two_l1805_180512

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils Serenity and her friends have -/
def total_pencils : ℕ := 21

/-- The number of color boxes Serenity bought -/
def serenity_boxes : ℕ := 1

/-- The number of Serenity's friends who bought the color box -/
def friends_with_boxes : ℕ := (total_pencils / pencils_per_box) - serenity_boxes

theorem friends_with_boxes_eq_two : friends_with_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_friends_with_boxes_eq_two_l1805_180512


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l1805_180538

/-- Proves that the employee discount percentage is 10% given the problem conditions --/
theorem employee_discount_percentage 
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l1805_180538


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1805_180536

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1805_180536


namespace NUMINAMATH_CALUDE_ellipse_k_value_l1805_180511

/-- The equation of an ellipse with a parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + (k*y^2)/5 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: For an ellipse with the given equation and focus, k equals 1 -/
theorem ellipse_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, ellipse_equation x y k) ∧ 
  (focus.1 = 0 ∧ focus.2 = 2) → k = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l1805_180511


namespace NUMINAMATH_CALUDE_line_transformation_l1805_180595

/-- Given a line l: ax + y - 7 = 0 transformed by matrix A to line l': 9x + y - 91 = 0,
    prove that a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + y - 7 = 0 → 
    9 * (3 * x) + (-x + b * y) - 91 = 0) → 
  a = 2 ∧ b = 13 := by
sorry

end NUMINAMATH_CALUDE_line_transformation_l1805_180595


namespace NUMINAMATH_CALUDE_total_with_tax_calculation_l1805_180531

def total_before_tax : ℝ := 150
def sales_tax_rate : ℝ := 0.08

theorem total_with_tax_calculation :
  total_before_tax * (1 + sales_tax_rate) = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_with_tax_calculation_l1805_180531


namespace NUMINAMATH_CALUDE_triangle_equilateral_conditions_l1805_180529

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h_a h_b h_c : ℝ)
  (ha_pos : h_a > 0)
  (hb_pos : h_b > 0)
  (hc_pos : h_c > 0)

-- Define the property of having equal sums of side and height
def equal_side_height_sums (t : Triangle) : Prop :=
  t.a + t.h_a = t.b + t.h_b ∧ t.b + t.h_b = t.c + t.h_c

-- Define the property of having equal inscribed squares
def equal_inscribed_squares (t : Triangle) : Prop :=
  (2 * t.a * t.h_a) / (t.a + t.h_a) = (2 * t.b * t.h_b) / (t.b + t.h_b) ∧
  (2 * t.b * t.h_b) / (t.b + t.h_b) = (2 * t.c * t.h_c) / (t.c + t.h_c)

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_equilateral_conditions (t : Triangle) :
  (equal_side_height_sums t ∨ equal_inscribed_squares t) → is_equilateral t :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_conditions_l1805_180529


namespace NUMINAMATH_CALUDE_product_remainder_l1805_180555

theorem product_remainder (a b m : ℕ) (h : a * b = 145 * 155) (hm : m = 12) : 
  (a * b) % m = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1805_180555


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_l1805_180556

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ 4 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_l1805_180556


namespace NUMINAMATH_CALUDE_real_roots_condition_zero_sum_of_squares_l1805_180534

-- Statement 1
theorem real_roots_condition (q : ℝ) :
  q < 1 → ∃ x : ℝ, x^2 + 2*x + q = 0 :=
sorry

-- Statement 2
theorem zero_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 0 → x = 0 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_zero_sum_of_squares_l1805_180534


namespace NUMINAMATH_CALUDE_candies_left_theorem_l1805_180569

/-- Calculates the number of candies left to be shared with others --/
def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (candies_to_eat : ℕ) : ℕ :=
  let candies_after_siblings := initial_candies - siblings * candies_per_sibling
  let candies_after_friend := candies_after_siblings / 2
  candies_after_friend - candies_to_eat

/-- Proves that given the initial conditions, the number of candies left to be shared with others is 19 --/
theorem candies_left_theorem :
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_theorem_l1805_180569


namespace NUMINAMATH_CALUDE_prime_fraction_sum_l1805_180540

theorem prime_fraction_sum (p q x y : ℕ) : 
  Prime p → Prime q → x > 0 → y > 0 → x < p → y < q → 
  (∃ k : ℤ, (p : ℚ) / x + (q : ℚ) / y = k) → x = y := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_sum_l1805_180540


namespace NUMINAMATH_CALUDE_problem_solution_l1805_180575

theorem problem_solution : 
  (1 - 1^2022 - (3 * (2/3)^2 - 8/3 / (-2)^3) = -8/3) ∧ 
  (2^3 / 3 * (-1/4 + 7/12 - 5/6) / (-1/18) = 24) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1805_180575


namespace NUMINAMATH_CALUDE_meeting_speed_l1805_180508

/-- Given two people 55 miles apart, where one walks at 6 mph and the other walks 25 miles before they meet, prove that the speed of the second person is 5 mph. -/
theorem meeting_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 55 →
  fred_speed = 6 →
  sam_distance = 25 →
  (total_distance - sam_distance) / fred_speed = sam_distance / ((total_distance - sam_distance) / fred_speed) :=
by sorry

end NUMINAMATH_CALUDE_meeting_speed_l1805_180508


namespace NUMINAMATH_CALUDE_intersection_dot_product_l1805_180578

/-- Given an ellipse and a hyperbola with common foci, the dot product of vectors from their intersection point to the foci is 21. -/
theorem intersection_dot_product (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  x^2/25 + y^2/16 = 1 →  -- Ellipse equation
  x^2/4 - y^2/5 = 1 →    -- Hyperbola equation
  P = (x, y) →           -- P is on both curves
  (∃ c : ℝ, c > 0 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (5 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (5 - c)^2 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (2 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (c - 2)^2) →  -- Common foci condition
  ((F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) : ℝ) = 21 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l1805_180578


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1805_180584

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ)
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = termite_ridden * 5 / 8) :
  (termite_ridden - collapsing : ℚ) / total_homes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1805_180584


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l1805_180537

theorem trig_identity_simplification (x y : ℝ) : 
  Real.cos (x + y) * Real.sin y - Real.sin (x + y) * Real.cos y = -Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l1805_180537


namespace NUMINAMATH_CALUDE_set_equivalence_l1805_180566

theorem set_equivalence (U A B : Set α) :
  (A ∩ B = A) ↔ (A ⊆ U ∧ B ⊆ U ∧ (Uᶜ ∩ B)ᶜ ⊆ (Uᶜ ∩ A)ᶜ) := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l1805_180566


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1805_180586

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) →
  (10 * x₂^2 + 15 * x₂ - 20 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1805_180586


namespace NUMINAMATH_CALUDE_lcm_1332_888_l1805_180563

theorem lcm_1332_888 : Nat.lcm 1332 888 = 2664 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1332_888_l1805_180563


namespace NUMINAMATH_CALUDE_vector_norm_equation_solution_l1805_180514

theorem vector_norm_equation_solution :
  let v : ℝ × ℝ := (3, -2)
  let w : ℝ × ℝ := (6, -1)
  let norm_squared (x : ℝ × ℝ) := x.1^2 + x.2^2
  { k : ℝ | norm_squared (k * v.1 - w.1, k * v.2 - w.2) = 34 } = {3, 1/13} := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_equation_solution_l1805_180514


namespace NUMINAMATH_CALUDE_card_collection_problem_l1805_180589

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of squares of the first n natural numbers -/
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The average value of cards in a collection where each number k from 1 to n appears k times -/
def average_card_value (n : ℕ) : ℚ :=
  (sum_squares_first_n n : ℚ) / (sum_first_n n : ℚ)

theorem card_collection_problem :
  ∃ m : ℕ, average_card_value m = 56 ∧ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_problem_l1805_180589


namespace NUMINAMATH_CALUDE_halfway_fraction_l1805_180547

theorem halfway_fraction (a b : ℚ) (ha : a = 1/6) (hb : b = 1/4) :
  (a + b) / 2 = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1805_180547


namespace NUMINAMATH_CALUDE_old_man_coins_l1805_180567

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_old_man_coins_l1805_180567


namespace NUMINAMATH_CALUDE_negation_statement_is_false_l1805_180519

theorem negation_statement_is_false : ¬(
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x)
) := by sorry

end NUMINAMATH_CALUDE_negation_statement_is_false_l1805_180519


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1805_180532

/-- If the cost price of 29 articles equals the selling price of 24 articles,
    then the percentage of profit is 5/24 * 100. -/
theorem merchant_profit_percentage (C S : ℝ) (h : 29 * C = 24 * S) :
  (S - C) / C * 100 = 5 / 24 * 100 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1805_180532


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1805_180574

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b+1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1805_180574


namespace NUMINAMATH_CALUDE_horner_rule_v3_l1805_180503

def horner_rule (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_rule_v3 (x : ℝ) (h : x = 2) :
  let a := [1, 2, 0, -1, 3]
  let v₃ := horner_rule (a.take 4) x
  v₃ = 20 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_l1805_180503


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1805_180591

theorem polar_to_cartesian (x y : ℝ) : 
  (∃ (ρ θ : ℝ), ρ = 3 ∧ θ = π/6 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → 
  x = 3 * Real.sqrt 3 / 2 ∧ y = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1805_180591


namespace NUMINAMATH_CALUDE_fundamental_disagreement_essence_l1805_180502

-- Define philosophical viewpoints
def materialist_viewpoint : String := "Without scenery, where does emotion come from?"
def idealist_viewpoint : String := "Without emotion, where does scenery come from?"

-- Define the concept of fundamental disagreement
def fundamental_disagreement (v1 v2 : String) : Prop := sorry

-- Define the essence of the world
inductive WorldEssence
| Material
| Consciousness

-- Theorem statement
theorem fundamental_disagreement_essence :
  fundamental_disagreement materialist_viewpoint idealist_viewpoint ↔
  ∃ (e : WorldEssence), (e = WorldEssence.Material ∨ e = WorldEssence.Consciousness) :=
sorry

end NUMINAMATH_CALUDE_fundamental_disagreement_essence_l1805_180502


namespace NUMINAMATH_CALUDE_science_club_membership_l1805_180554

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : biology = 50)
  (h3 : chemistry = 40)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_science_club_membership_l1805_180554


namespace NUMINAMATH_CALUDE_qingming_festival_probability_l1805_180521

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade7 : ℕ
  grade8 : ℕ
  grade9 : ℕ

/-- Represents the participation methods for each grade --/
structure ParticipationMethods where
  memorial_hall : GradeDistribution
  online : GradeDistribution

/-- The main theorem to prove --/
theorem qingming_festival_probability 
  (total_students : GradeDistribution)
  (participation : ParticipationMethods)
  (h1 : total_students.grade7 = 4 * k)
  (h2 : total_students.grade8 = 5 * k)
  (h3 : total_students.grade9 = 6 * k)
  (h4 : participation.memorial_hall.grade7 = 2 * a - 1)
  (h5 : participation.memorial_hall.grade8 = 8)
  (h6 : participation.memorial_hall.grade9 = 10)
  (h7 : participation.online.grade7 = a)
  (h8 : participation.online.grade8 = b)
  (h9 : participation.online.grade9 = 2)
  (h10 : total_students.grade7 = participation.memorial_hall.grade7 + participation.online.grade7)
  (h11 : total_students.grade8 = participation.memorial_hall.grade8 + participation.online.grade8)
  (h12 : total_students.grade9 = participation.memorial_hall.grade9 + participation.online.grade9)
  : ℚ :=
  5/21

/-- Auxiliary function to calculate combinations --/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

#check qingming_festival_probability

end NUMINAMATH_CALUDE_qingming_festival_probability_l1805_180521


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1805_180527

theorem geometric_sequence_first_term (a b c d : ℚ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    a * r^4 = 48 ∧ 
    a * r^5 = 192) →
  a = 3/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1805_180527


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1805_180592

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 5, 7}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 7} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1805_180592


namespace NUMINAMATH_CALUDE_first_train_crossing_time_l1805_180543

/-- Two trains running in opposite directions with equal speeds -/
structure TwoTrains where
  v₁ : ℝ  -- Speed of the first train
  v₂ : ℝ  -- Speed of the second train
  L₁ : ℝ  -- Length of the first train
  L₂ : ℝ  -- Length of the second train
  t₂ : ℝ  -- Time taken by the second train to cross the man
  cross_time : ℝ  -- Time taken for the trains to cross each other

/-- The conditions given in the problem -/
def problem_conditions (trains : TwoTrains) : Prop :=
  trains.v₁ > 0 ∧ 
  trains.v₂ > 0 ∧ 
  trains.L₁ > 0 ∧ 
  trains.L₂ > 0 ∧ 
  trains.v₁ = trains.v₂ ∧  -- Ratio of speeds is 1
  trains.t₂ = 17 ∧  -- Second train crosses the man in 17 seconds
  trains.cross_time = 22 ∧  -- Trains cross each other in 22 seconds
  (trains.L₁ + trains.L₂) / (trains.v₁ + trains.v₂) = trains.cross_time

/-- The theorem to be proved -/
theorem first_train_crossing_time (trains : TwoTrains) 
  (h : problem_conditions trains) : 
  trains.L₁ / trains.v₁ = 27 := by
  sorry


end NUMINAMATH_CALUDE_first_train_crossing_time_l1805_180543


namespace NUMINAMATH_CALUDE_man_crossing_bridge_l1805_180583

/-- Proves that a man walking at 10 km/hr takes 10 minutes to cross a 1666.6666666666665 meter bridge -/
theorem man_crossing_bridge 
  (walking_rate : ℝ) 
  (bridge_length : ℝ) 
  (h1 : walking_rate = 10) -- km/hr
  (h2 : bridge_length = 1666.6666666666665) -- meters
  : (bridge_length / (walking_rate * 1000 / 60)) = 10 := by
  sorry

#check man_crossing_bridge

end NUMINAMATH_CALUDE_man_crossing_bridge_l1805_180583


namespace NUMINAMATH_CALUDE_keaton_apple_harvest_interval_l1805_180599

/-- Represents Keaton's farm and harvesting schedule -/
structure Farm where
  orange_harvest_interval : ℕ  -- months between orange harvests
  orange_harvest_value : ℕ     -- value of each orange harvest in dollars
  apple_harvest_value : ℕ      -- value of each apple harvest in dollars
  total_yearly_earnings : ℕ    -- total earnings per year in dollars

/-- Calculates how often Keaton can harvest his apples -/
def apple_harvest_interval (f : Farm) : ℕ :=
  12 / ((f.total_yearly_earnings - (12 / f.orange_harvest_interval * f.orange_harvest_value)) / f.apple_harvest_value)

/-- Theorem stating that Keaton can harvest his apples every 3 months -/
theorem keaton_apple_harvest_interval :
  ∀ (f : Farm),
  f.orange_harvest_interval = 2 →
  f.orange_harvest_value = 50 →
  f.apple_harvest_value = 30 →
  f.total_yearly_earnings = 420 →
  apple_harvest_interval f = 3 := by
  sorry

end NUMINAMATH_CALUDE_keaton_apple_harvest_interval_l1805_180599


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1805_180588

/-- The equation (x+5)(x+2) = m + 3x has exactly one real solution if and only if m = 6 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1805_180588


namespace NUMINAMATH_CALUDE_divisibility_property_l1805_180524

theorem divisibility_property (p : ℕ) (h_odd : Odd p) (h_gt_one : p > 1) :
  ∃ k : ℤ, (p - 1) ^ ((p - 1) / 2) - 1 = (p - 2) * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l1805_180524


namespace NUMINAMATH_CALUDE_orange_savings_l1805_180515

theorem orange_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) :
  liam_oranges = 40 →
  liam_price = 5/2 →
  claire_oranges = 30 →
  claire_price = 6/5 →
  (liam_oranges / 2 * liam_price + claire_oranges * claire_price : ℚ) = 86 := by
  sorry

end NUMINAMATH_CALUDE_orange_savings_l1805_180515


namespace NUMINAMATH_CALUDE_complex_equation_l1805_180522

/-- Given x ∈ ℝ, y is a pure imaginary number, and (x-y)i = 2-i, then x+y = -1+2i -/
theorem complex_equation (x : ℝ) (y : ℂ) (h1 : y.re = 0) (h2 : (x - y) * Complex.I = 2 - Complex.I) : 
  x + y = -1 + 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_l1805_180522


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l1805_180573

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ (s₁ s₂ : ℝ), 
    s₁^2 = x^2 + 12*x + 36 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 64) → 
  x = 13/3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l1805_180573


namespace NUMINAMATH_CALUDE_range_of_k_with_two_preimages_l1805_180564

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem range_of_k_with_two_preimages :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = k ∧ f y = k) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_with_two_preimages_l1805_180564


namespace NUMINAMATH_CALUDE_rods_per_sheet_is_correct_l1805_180570

/-- Represents the number of metal rods in each metal sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 2

/-- Represents the total number of fence panels -/
def total_panels : ℕ := 10

/-- Represents the number of metal rods in each metal beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the entire fence -/
def total_rods : ℕ := 380

/-- Theorem stating that the number of rods per sheet is correct given the conditions -/
theorem rods_per_sheet_is_correct :
  rods_per_sheet * (sheets_per_panel * total_panels) + 
  rods_per_beam * (beams_per_panel * total_panels) = total_rods :=
by sorry

end NUMINAMATH_CALUDE_rods_per_sheet_is_correct_l1805_180570


namespace NUMINAMATH_CALUDE_parabola_focus_l1805_180541

/-- The parabola equation --/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola --/
def focus (p q : ℝ) : Prop := p = 0 ∧ q = 2

theorem parabola_focus :
  ∀ x y : ℝ, parabola x y → ∃ p q : ℝ, focus p q := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1805_180541


namespace NUMINAMATH_CALUDE_triangle_angle_extension_l1805_180504

theorem triangle_angle_extension (a b c x : Real) : 
  a = 50 → b = 60 → c = 180 - a - b → 
  x = 180 - (180 - c) → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_extension_l1805_180504


namespace NUMINAMATH_CALUDE_cricket_team_handedness_l1805_180593

theorem cricket_team_handedness (total_players : Nat) (throwers : Nat) (right_handed : Nat)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : right_handed = 51)
    (h4 : throwers ≤ right_handed) :
    (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_handedness_l1805_180593


namespace NUMINAMATH_CALUDE_solution_range_l1805_180550

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ equation x m) ↔ (m ≥ -5 ∧ m ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l1805_180550


namespace NUMINAMATH_CALUDE_mean_value_point_of_cubic_minus_linear_l1805_180594

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3*x^2 - 3

-- Define the mean value point property
def is_mean_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b x₀ : ℝ) : Prop :=
  f b - f a = f' x₀ * (b - a)

theorem mean_value_point_of_cubic_minus_linear :
  ∃ x₀ : ℝ, is_mean_value_point f f' (-2) 2 x₀ ∧ x₀^2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_mean_value_point_of_cubic_minus_linear_l1805_180594


namespace NUMINAMATH_CALUDE_book_selling_price_l1805_180549

theorem book_selling_price 
  (num_books : ℕ) 
  (buying_price : ℚ) 
  (price_difference : ℚ) 
  (h1 : num_books = 15)
  (h2 : buying_price = 11)
  (h3 : price_difference = 210) :
  ∃ (selling_price : ℚ), 
    selling_price * num_books - buying_price * num_books = price_difference ∧ 
    selling_price = 25 :=
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l1805_180549


namespace NUMINAMATH_CALUDE_triangle_unique_solution_l1805_180500

open Real

theorem triangle_unique_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! B : ℝ, 0 < B ∧ B < π ∧ sin B = (b / a) * sin A :=
sorry

end NUMINAMATH_CALUDE_triangle_unique_solution_l1805_180500


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1805_180572

def vector_a (t : ℝ) : Fin 2 → ℝ := ![t, 1]
def vector_b : Fin 2 → ℝ := ![2, 4]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem perpendicular_vectors (t : ℝ) :
  perpendicular (vector_a t) vector_b → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1805_180572


namespace NUMINAMATH_CALUDE_sandy_grew_six_carrots_l1805_180516

/-- The number of carrots grown by Sandy -/
def sandy_carrots : ℕ := sorry

/-- The number of carrots grown by Sam -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown by Sandy and Sam -/
def total_carrots : ℕ := 9

/-- Theorem stating that Sandy grew 6 carrots -/
theorem sandy_grew_six_carrots : sandy_carrots = 6 := by sorry

end NUMINAMATH_CALUDE_sandy_grew_six_carrots_l1805_180516


namespace NUMINAMATH_CALUDE_lisa_flight_distance_l1805_180557

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Lisa's flight distance -/
theorem lisa_flight_distance :
  let speed : ℝ := 32
  let time : ℝ := 8
  distance speed time = 256 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_distance_l1805_180557


namespace NUMINAMATH_CALUDE_number_problem_l1805_180505

theorem number_problem (x : ℝ) : 4 * x = 166.08 → (x / 4) + 0.48 = 10.86 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1805_180505


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1805_180551

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ max) ∧
  (x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ 961/8) :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1805_180551


namespace NUMINAMATH_CALUDE_triangle_centroid_existence_and_property_l1805_180576

/-- Given a triangle ABC, there exists a unique point O (the centroid) that lies on all medians and divides each in a 2:1 ratio from the vertex. -/
theorem triangle_centroid_existence_and_property (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃! O : EuclideanSpace ℝ (Fin 2),
    (∃ t : ℝ, O = A + t • (midpoint ℝ B C - A)) ∧
    (∃ u : ℝ, O = B + u • (midpoint ℝ A C - B)) ∧
    (∃ v : ℝ, O = C + v • (midpoint ℝ A B - C)) ∧
    (O = A + (2/3) • (midpoint ℝ B C - A)) ∧
    (O = B + (2/3) • (midpoint ℝ A C - B)) ∧
    (O = C + (2/3) • (midpoint ℝ A B - C)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_existence_and_property_l1805_180576


namespace NUMINAMATH_CALUDE_three_witnesses_are_liars_l1805_180558

-- Define the type for witnesses
inductive Witness : Type
  | one
  | two
  | three
  | four

-- Define a function to represent the statement of each witness
def statement (w : Witness) : Nat :=
  match w with
  | Witness.one => 1
  | Witness.two => 2
  | Witness.three => 3
  | Witness.four => 4

-- Define a predicate to check if a witness is telling the truth
def isTruthful (w : Witness) (numLiars : Nat) : Prop :=
  statement w = numLiars

-- Theorem: Exactly three witnesses are liars
theorem three_witnesses_are_liars :
  ∃! (numLiars : Nat), 
    numLiars = 3 ∧
    (∃! (truthful : Witness), 
      isTruthful truthful numLiars ∧
      ∀ (w : Witness), w ≠ truthful → ¬(isTruthful w numLiars)) :=
by
  sorry


end NUMINAMATH_CALUDE_three_witnesses_are_liars_l1805_180558


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1805_180533

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + I) / I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1805_180533


namespace NUMINAMATH_CALUDE_calculation_proof_l1805_180582

theorem calculation_proof : 10 - 9 * 8 / 4 + 7 - 6 * 5 + 3 - 2 * 1 = -30 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1805_180582


namespace NUMINAMATH_CALUDE_exist_three_distinct_naturals_sum_product_squares_l1805_180535

theorem exist_three_distinct_naturals_sum_product_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (m : ℕ), a + b + c = m^2) ∧
  (∃ (n : ℕ), a * b * c = n^2) := by
  sorry

end NUMINAMATH_CALUDE_exist_three_distinct_naturals_sum_product_squares_l1805_180535


namespace NUMINAMATH_CALUDE_central_region_area_l1805_180565

/-- The area of the central region in a square with intersecting lines --/
theorem central_region_area (s : ℝ) (h : s = 10) : 
  let a := s / 3
  let b := 2 * s / 3
  let central_side := (s - (a + b)) / 2
  central_side ^ 2 = (s / 6) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_central_region_area_l1805_180565


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1805_180545

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with one face painted in a rectangular solid -/
def cubes_with_one_face_painted (solid : RectangularSolid) : ℕ :=
  2 * ((solid.length - 2) * (solid.width - 2) +
       (solid.length - 2) * (solid.height - 2) +
       (solid.width - 2) * (solid.height - 2))

/-- Theorem: In a 9x10x11 rectangular solid, 382 cubes have exactly one face painted -/
theorem painted_cubes_count :
  let solid : RectangularSolid := ⟨9, 10, 11⟩
  cubes_with_one_face_painted solid = 382 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1805_180545


namespace NUMINAMATH_CALUDE_equation_solutions_l1805_180528

theorem equation_solutions :
  (∃ x : ℚ, x / (3/4) = 2 / (9/10) ∧ x = 5/3) ∧
  (∃ x : ℚ, 0.5 / x = 0.75 / 6 ∧ x = 4) ∧
  (∃ x : ℚ, x / 20 = 2/5 ∧ x = 8) ∧
  (∃ x : ℚ, (3/4 * x) / 15 = 2/3 ∧ x = 40/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1805_180528


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l1805_180530

/-- Calculates the distance a boat travels along a stream in one hour, given its speed in still water and its distance against the stream in one hour. -/
def distance_along_stream (boat_speed : ℝ) (distance_against : ℝ) : ℝ :=
  let stream_speed := boat_speed - distance_against
  boat_speed + stream_speed

/-- Theorem stating that a boat with a speed of 7 km/hr in still water,
    traveling 3 km against the stream in one hour,
    will travel 11 km along the stream in one hour. -/
theorem boat_distance_theorem :
  distance_along_stream 7 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_theorem_l1805_180530


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_is_half_more_than_forty_l1805_180506

theorem twenty_five_percent_less_than_eighty_is_half_more_than_forty : 
  ∃ x : ℝ, (80 - 0.25 * 80 = x + 0.5 * x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_is_half_more_than_forty_l1805_180506


namespace NUMINAMATH_CALUDE_kareem_largest_l1805_180579

def jose_calc (x : Int) : Int :=
  ((x - 2) * 3) + 5

def thuy_calc (x : Int) : Int :=
  (x * 3 - 2) + 5

def kareem_calc (x : Int) : Int :=
  ((x - 2) + 5) * 3

theorem kareem_largest (start : Int) :
  start = 15 →
  kareem_calc start > jose_calc start ∧
  kareem_calc start > thuy_calc start :=
by
  sorry

#eval jose_calc 15
#eval thuy_calc 15
#eval kareem_calc 15

end NUMINAMATH_CALUDE_kareem_largest_l1805_180579


namespace NUMINAMATH_CALUDE_original_price_after_discount_l1805_180597

theorem original_price_after_discount (a : ℝ) (h : a > 0) : 
  (4/5 : ℝ) * ((5/4 : ℝ) * a) = a := by sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l1805_180597


namespace NUMINAMATH_CALUDE_factorial_expression_is_perfect_square_l1805_180509

theorem factorial_expression_is_perfect_square (n : ℕ) (h : n ≥ 10) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial (n + 1) = (n + 2) ^ 2 := by
  sorry

#check factorial_expression_is_perfect_square

end NUMINAMATH_CALUDE_factorial_expression_is_perfect_square_l1805_180509


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l1805_180587

/-- A parabola is defined by its equation in the form y² = -4px, where p is the focal length. -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -4 * p * x

/-- The directrix of a parabola is a vertical line with equation x = p. -/
def directrix (parabola : Parabola) : ℝ → Prop :=
  fun x => x = parabola.p

theorem parabola_directrix_equation :
  ∀ (y : ℝ), ∃ (parabola : Parabola),
    (∀ (x : ℝ), parabola.equation x y ↔ y^2 = -4*x) →
    (∀ (x : ℝ), directrix parabola x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l1805_180587


namespace NUMINAMATH_CALUDE_quadratic_product_zero_l1805_180552

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if f((a - b - c)/(2a)) = 0 and f((c - a - b)/(2a)) = 0, 
    then f(-1) * f(1) = 0 -/
theorem quadratic_product_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f ((a - b - c) / (2 * a)) = 0)
  (h3 : f ((c - a - b) / (2 * a)) = 0)
  : f (-1) * f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_zero_l1805_180552


namespace NUMINAMATH_CALUDE_jungkook_has_bigger_number_l1805_180581

theorem jungkook_has_bigger_number : 
  let jungkook_number := 3 + 6
  let yoongi_number := 4
  jungkook_number > yoongi_number := by
sorry

end NUMINAMATH_CALUDE_jungkook_has_bigger_number_l1805_180581


namespace NUMINAMATH_CALUDE_hit_probability_theorem_l1805_180598

/-- The probability of hitting a target with one shot -/
def hit_probability : ℚ := 1 / 2

/-- The number of shots taken -/
def total_shots : ℕ := 6

/-- The number of hits required -/
def required_hits : ℕ := 3

/-- The number of consecutive hits required -/
def consecutive_hits : ℕ := 2

/-- The probability of hitting the target 3 times with exactly 2 consecutive hits out of 6 shots -/
def target_probability : ℚ := (Nat.choose 4 2 : ℚ) * (hit_probability ^ total_shots)

theorem hit_probability_theorem : 
  target_probability = (Nat.choose 4 2 : ℚ) * ((1 : ℚ) / 2) ^ 6 := by sorry

end NUMINAMATH_CALUDE_hit_probability_theorem_l1805_180598


namespace NUMINAMATH_CALUDE_closest_beetle_positions_l1805_180501

structure Table where
  sugar_position : ℝ × ℝ
  ant_radius : ℝ
  beetle_radius : ℝ
  ant_initial_position : ℝ × ℝ
  beetle_initial_position : ℝ × ℝ

def closest_positions (t : Table) : Set (ℝ × ℝ) :=
  {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)}

theorem closest_beetle_positions (t : Table) 
  (h1 : t.sugar_position = (0, 0))
  (h2 : t.ant_radius = 2)
  (h3 : t.beetle_radius = 4)
  (h4 : t.ant_initial_position = (-1, Real.sqrt 3))
  (h5 : t.beetle_initial_position = (2 * Real.sqrt 3, 2)) :
  closest_positions t = {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)} := by
  sorry

end NUMINAMATH_CALUDE_closest_beetle_positions_l1805_180501


namespace NUMINAMATH_CALUDE_average_seashells_per_person_l1805_180585

/-- The number of seashells found by Sally -/
def sally_shells : ℕ := 9

/-- The number of seashells found by Tom -/
def tom_shells : ℕ := 7

/-- The number of seashells found by Jessica -/
def jessica_shells : ℕ := 5

/-- The number of seashells found by Alex -/
def alex_shells : ℕ := 12

/-- The total number of people who found seashells -/
def total_people : ℕ := 4

/-- The average number of seashells found per person -/
def average_shells : ℚ := (sally_shells + tom_shells + jessica_shells + alex_shells : ℚ) / total_people

theorem average_seashells_per_person :
  average_shells = 33 / 4 :=
by sorry

end NUMINAMATH_CALUDE_average_seashells_per_person_l1805_180585


namespace NUMINAMATH_CALUDE_factorization_equality_l1805_180507

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1805_180507


namespace NUMINAMATH_CALUDE_room_width_proof_l1805_180596

theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  veranda_width = 2 →
  veranda_area = 144 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_proof_l1805_180596


namespace NUMINAMATH_CALUDE_sin_equality_proof_l1805_180510

theorem sin_equality_proof (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (-474 * π / 180) → n = 66 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l1805_180510


namespace NUMINAMATH_CALUDE_three_competition_participation_l1805_180561

theorem three_competition_participation 
  (total : ℕ) 
  (chinese : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (chinese_math : ℕ) 
  (math_english : ℕ) 
  (chinese_english : ℕ) 
  (none : ℕ) 
  (h1 : total = 100)
  (h2 : chinese = 39)
  (h3 : math = 49)
  (h4 : english = 41)
  (h5 : chinese_math = 14)
  (h6 : math_english = 13)
  (h7 : chinese_english = 9)
  (h8 : none = 1) :
  ∃ (all_three : ℕ), 
    all_three = 6 ∧ 
    total = chinese + math + english - chinese_math - math_english - chinese_english + all_three + none :=
by sorry

end NUMINAMATH_CALUDE_three_competition_participation_l1805_180561


namespace NUMINAMATH_CALUDE_student_council_committees_l1805_180523

theorem student_council_committees (n : ℕ) (k : ℕ) (m : ℕ) (p : ℕ) (w : ℕ) :
  n = 15 →  -- Total number of student council members
  k = 3 →   -- Size of welcoming committee
  m = 4 →   -- Size of planning committee
  p = 2 →   -- Size of finance committee
  w = 20 →  -- Number of ways to select welcoming committee
  (n.choose m) * (k.choose p) = 4095 :=
by sorry

end NUMINAMATH_CALUDE_student_council_committees_l1805_180523


namespace NUMINAMATH_CALUDE_intersection_A_B_l1805_180577

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- Define the interval [2, 3)
def interval_2_3 : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_2_3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1805_180577


namespace NUMINAMATH_CALUDE_exponential_characterization_l1805_180517

/-- A continuous function satisfying f(x+y) = f(x)f(y) is of the form aˣ for some a > 0 -/
theorem exponential_characterization (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_nonzero : ∃ x₀, f x₀ ≠ 0) 
  (h_mult : ∀ x y, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x, f x = Real.exp (x * Real.log a) := by
sorry

end NUMINAMATH_CALUDE_exponential_characterization_l1805_180517


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1805_180526

theorem quadratic_real_solutions : ∃ (x : ℝ), x^2 + 3*x - 2 = 0 ∧
  (∀ (x : ℝ), 2*x^2 - x + 1 ≠ 0) ∧
  (∀ (x : ℝ), x^2 - 2*x + 2 ≠ 0) ∧
  (∀ (x : ℝ), x^2 + 2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1805_180526


namespace NUMINAMATH_CALUDE_austin_picked_24_bags_l1805_180544

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of additional bags of apples Austin picked compared to Dallas -/
def austin_extra_apples : ℕ := 6

/-- The number of fewer bags of pears Austin picked compared to Dallas -/
def austin_fewer_pears : ℕ := 5

/-- The total number of bags of fruit Austin picked -/
def austin_total : ℕ := (dallas_apples + austin_extra_apples) + (dallas_pears - austin_fewer_pears)

theorem austin_picked_24_bags :
  austin_total = 24 := by sorry

end NUMINAMATH_CALUDE_austin_picked_24_bags_l1805_180544


namespace NUMINAMATH_CALUDE_factorization_equality_l1805_180568

theorem factorization_equality (a b : ℝ) : a^2 * b - 2*a*b + b = b * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1805_180568


namespace NUMINAMATH_CALUDE_books_sold_l1805_180525

/-- Proves the number of books Adam sold given initial count, books bought, and final count -/
theorem books_sold (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 33 → bought = 23 → final = 45 → initial - (initial - final + bought) = 11 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1805_180525


namespace NUMINAMATH_CALUDE_game_probability_l1805_180590

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Cindy : Player
| Dave : Player

/-- The game state is represented by a function from Player to ℕ (natural numbers) -/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has 2 units -/
def initialState : GameState :=
  fun _ => 2

/-- A single round of the game -/
def gameRound (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- The probability of a specific outcome after one round -/
def roundProbability (initialState finalState : GameState) : ℚ :=
  sorry -- Implementation details omitted

/-- The probability of all players having 2 units after 5 rounds -/
def finalProbability : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem stating the probability of all players having 2 units after 5 rounds -/
theorem game_probability : finalProbability = 4 / 81^5 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l1805_180590


namespace NUMINAMATH_CALUDE_kendra_sunday_shirts_l1805_180559

/-- The number of shirts Kendra wears in two weeks -/
def total_shirts : ℕ := 22

/-- The number of weekdays in two weeks -/
def weekdays : ℕ := 10

/-- The number of days Kendra changes shirts for after-school club in two weeks -/
def club_days : ℕ := 6

/-- The number of Saturdays in two weeks -/
def saturdays : ℕ := 2

/-- The number of Sundays in two weeks -/
def sundays : ℕ := 2

/-- The number of shirts Kendra wears on weekdays for school and club in two weeks -/
def weekday_shirts : ℕ := weekdays + club_days

/-- The number of shirts Kendra wears on Saturdays in two weeks -/
def saturday_shirts : ℕ := saturdays

theorem kendra_sunday_shirts :
  total_shirts - (weekday_shirts + saturday_shirts) = 4 := by
sorry

end NUMINAMATH_CALUDE_kendra_sunday_shirts_l1805_180559


namespace NUMINAMATH_CALUDE_prob_four_same_face_five_coins_l1805_180539

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least 'num_same' coins showing the same face when flipping 'num_coins' fair coins -/
def prob_same_face (num_same : ℕ) : ℚ :=
  let total_outcomes := 2^num_coins
  let favorable_outcomes := 2 * (Nat.choose num_coins (num_coins - num_same + 1))
  favorable_outcomes / total_outcomes

/-- The probability of getting at least 4 coins showing the same face when flipping 5 fair coins is 3/8 -/
theorem prob_four_same_face_five_coins : prob_same_face 4 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_same_face_five_coins_l1805_180539


namespace NUMINAMATH_CALUDE_altitude_division_ratio_l1805_180546

/-- Given a triangle with side lengths √3, 2, and √5, the altitude perpendicular 
    to the side of length 2 divides that side in the ratio 1:3 -/
theorem altitude_division_ratio (a b c : ℝ) (h₁ : a = Real.sqrt 3) 
    (h₂ : b = 2) (h₃ : c = Real.sqrt 5) :
    let m := Real.sqrt (3 - (1/2)^2)
    (1/2) / (3/2) = 1/3 := by sorry

end NUMINAMATH_CALUDE_altitude_division_ratio_l1805_180546


namespace NUMINAMATH_CALUDE_unique_prime_power_condition_l1805_180571

theorem unique_prime_power_condition : ∃! p : ℕ, 
  p ≤ 1000 ∧ 
  Nat.Prime p ∧ 
  ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m ^ n ∧
  p = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_power_condition_l1805_180571


namespace NUMINAMATH_CALUDE_janice_purchase_l1805_180518

theorem janice_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 150 * y + 300 * z = 4500 →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_janice_purchase_l1805_180518


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1805_180562

open Real

/-- A function f : ℝ → ℝ is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem neither_sufficient_nor_necessary
  (a : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1) :
  ¬(IsIncreasing (fun x ↦ a^x) → IsIncreasing (fun x ↦ x^a)) ∧
  ¬(IsIncreasing (fun x ↦ x^a) → IsIncreasing (fun x ↦ a^x)) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1805_180562


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l1805_180580

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sequence_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the specific arithmetic sequence is 18599100 -/
theorem specific_arithmetic_sequence_sum :
  arithmetic_sequence_sum 2008 (-1776) 11 = 18599100 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l1805_180580


namespace NUMINAMATH_CALUDE_record_listening_time_l1805_180553

/-- The number of days required to listen to a record collection --/
def days_to_listen (initial_records : ℕ) (gift_records : ℕ) (purchased_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + purchased_records) * days_per_record

/-- Theorem: Given the initial conditions, it takes 100 days to listen to the entire record collection --/
theorem record_listening_time : days_to_listen 8 12 30 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_record_listening_time_l1805_180553


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1805_180520

theorem binomial_coefficient_n_minus_two (n : ℕ) (hn : n > 0) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1805_180520


namespace NUMINAMATH_CALUDE_problem_solution_l1805_180513

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ = 2)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ = 15)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ = 130) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ = 347 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1805_180513


namespace NUMINAMATH_CALUDE_max_games_buyable_l1805_180542

def total_earnings : ℝ := 180
def blade_percentage : ℝ := 0.35
def game_cost : ℝ := 12.50
def tax_rate : ℝ := 0.05

def remaining_money : ℝ := total_earnings * (1 - blade_percentage)
def game_cost_with_tax : ℝ := game_cost * (1 + tax_rate)

theorem max_games_buyable : 
  ⌊remaining_money / game_cost_with_tax⌋ = 8 :=
sorry

end NUMINAMATH_CALUDE_max_games_buyable_l1805_180542
