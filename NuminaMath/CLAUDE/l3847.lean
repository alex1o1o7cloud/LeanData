import Mathlib

namespace NUMINAMATH_CALUDE_circle_line_intersection_l3847_384718

theorem circle_line_intersection (α β : ℝ) (n k : ℤ) : 
  (∃ A B : ℝ × ℝ, 
    A = (Real.cos (2 * α), Real.cos (2 * β)) ∧ 
    B = (Real.cos (2 * β), Real.cos α) ∧
    (A = (-1/2, 0) ∧ B = (0, -1/2) ∨ A = (0, -1/2) ∧ B = (-1/2, 0))) →
  (α = 2 * Real.pi / 3 + 2 * Real.pi * ↑n ∨ 
   α = -2 * Real.pi / 3 + 2 * Real.pi * ↑n) ∧
  β = Real.pi / 4 + Real.pi / 2 * ↑k :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3847_384718


namespace NUMINAMATH_CALUDE_distance_AB_is_360_l3847_384701

/-- The distance between two points A and B --/
def distance_AB : ℝ := sorry

/-- The initial speed of the passenger train --/
def v_pass : ℝ := sorry

/-- The initial speed of the freight train --/
def v_freight : ℝ := sorry

/-- The time taken by the freight train to travel from A to B --/
def t_freight : ℝ := sorry

/-- The time difference between the passenger and freight trains --/
def time_diff : ℝ := 3.2

/-- The additional distance traveled by the passenger train --/
def additional_distance : ℝ := 288

/-- The speed increase for both trains --/
def speed_increase : ℝ := 10

/-- The new time difference after speed increase --/
def new_time_diff : ℝ := 2.4

theorem distance_AB_is_360 :
  v_pass * (t_freight - time_diff) = v_freight * t_freight + additional_distance ∧
  distance_AB / (v_freight + speed_increase) - distance_AB / (v_pass + speed_increase) = new_time_diff ∧
  distance_AB = v_freight * t_freight →
  distance_AB = 360 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_360_l3847_384701


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3847_384713

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time : 
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let train_length : ℝ := 120 -- meters

  let jogger_speed_ms : ℝ := jogger_speed * 1000 / 3600 -- Convert to m/s
  let train_speed_ms : ℝ := train_speed * 1000 / 3600 -- Convert to m/s
  let relative_speed : ℝ := train_speed_ms - jogger_speed_ms
  let total_distance : ℝ := initial_distance + train_length
  let time : ℝ := total_distance / relative_speed

  time = 36 := by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3847_384713


namespace NUMINAMATH_CALUDE_morgan_change_is_eleven_l3847_384754

/-- The change Morgan receives after buying lunch -/
def morgan_change (hamburger_cost onion_rings_cost smoothie_cost bill_amount : ℕ) : ℕ :=
  bill_amount - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change -/
theorem morgan_change_is_eleven :
  morgan_change 4 2 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_morgan_change_is_eleven_l3847_384754


namespace NUMINAMATH_CALUDE_root_implies_k_value_l3847_384735

theorem root_implies_k_value (k : ℝ) : 
  (2 * 7^2 + 3 * 7 - k = 0) → k = 119 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l3847_384735


namespace NUMINAMATH_CALUDE_traffic_light_probability_l3847_384749

theorem traffic_light_probability : 
  let p_A : ℚ := 25 / 60
  let p_B : ℚ := 35 / 60
  let p_C : ℚ := 45 / 60
  p_A * p_B * p_C = 35 / 192 := by
sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l3847_384749


namespace NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l3847_384752

/-- Represents the weights of family members and their relationships -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ
  total_weight : grandmother + daughter + child = 130
  daughter_child_weight : daughter + child = 60
  daughter_weight : daughter = 46

/-- The ratio of the child's weight to the grandmother's weight is 1:5 -/
theorem child_grandmother_weight_ratio (fw : FamilyWeights) :
  fw.child / fw.grandmother = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l3847_384752


namespace NUMINAMATH_CALUDE_terry_commute_time_l3847_384765

/-- Calculates the total daily driving time for Terry's commute -/
theorem terry_commute_time : 
  let segment1_distance : ℝ := 15
  let segment1_speed : ℝ := 30
  let segment2_distance : ℝ := 35
  let segment2_speed : ℝ := 50
  let segment3_distance : ℝ := 10
  let segment3_speed : ℝ := 40
  let total_time := 
    (segment1_distance / segment1_speed + 
     segment2_distance / segment2_speed + 
     segment3_distance / segment3_speed) * 2
  total_time = 2.9 := by sorry

end NUMINAMATH_CALUDE_terry_commute_time_l3847_384765


namespace NUMINAMATH_CALUDE_police_force_ratio_l3847_384708

/-- Given a police force with the following properties:
  * 20% of female officers were on duty
  * 100 officers were on duty that night
  * The police force has 250 female officers
  Prove that the ratio of female officers to total officers on duty is 1:2 -/
theorem police_force_ratio : 
  ∀ (total_female : ℕ) (on_duty : ℕ) (female_percent : ℚ),
  total_female = 250 →
  on_duty = 100 →
  female_percent = 1/5 →
  (female_percent * total_female) / on_duty = 1/2 := by
sorry

end NUMINAMATH_CALUDE_police_force_ratio_l3847_384708


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3847_384702

theorem triangle_angle_measure (a b : ℝ) (A B : ℝ) :
  a > 0 → b > 0 → 0 < A → A < π → 0 < B → B < π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3847_384702


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3847_384777

/-- The vertex of a quadratic function -/
theorem quadratic_vertex
  (a k c d : ℝ)
  (ha : a > 0)
  (hk : k ≠ b)  -- Note: 'b' is not defined, but kept as per the original problem
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ a * x^2 + k * x + c + d) :
  let x₀ := -k / (2 * a)
  ∃ y₀, (x₀, y₀) = (-k / (2 * a), -k^2 / (4 * a) + c + d) ∧ 
       ∀ x, f x ≥ f x₀ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3847_384777


namespace NUMINAMATH_CALUDE_closest_ratio_to_one_l3847_384768

/-- Represents the number of adults and children attending an exhibition -/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Calculates the total admission fee for a given attendance -/
def totalFee (a : Attendance) : ℕ :=
  25 * a.adults + 15 * a.children

/-- Checks if the ratio of adults to children is closer to 1 than the given ratio -/
def isCloserToOne (a : Attendance) (ratio : Rat) : Prop :=
  |1 - (a.adults : ℚ) / a.children| < |1 - ratio|

/-- The main theorem stating the closest ratio to 1 -/
theorem closest_ratio_to_one :
  ∃ (a : Attendance),
    a.adults > 0 ∧
    a.children > 0 ∧
    totalFee a = 1950 ∧
    a.adults = 48 ∧
    a.children = 50 ∧
    ∀ (b : Attendance),
      b.adults > 0 →
      b.children > 0 →
      totalFee b = 1950 →
      b ≠ a →
      isCloserToOne a (24 / 25) :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_to_one_l3847_384768


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3847_384796

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if 8a_2 + a_5 = 0, then S_3 / a_3 = 3/4 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →
  (8 * (a 2) + (a 5) = 0) →
  (S 3) / (a 3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3847_384796


namespace NUMINAMATH_CALUDE_first_cat_brown_eyed_kittens_l3847_384711

theorem first_cat_brown_eyed_kittens :
  ∀ (brown_eyed_first : ℕ),
  let blue_eyed_first : ℕ := 3
  let blue_eyed_second : ℕ := 4
  let brown_eyed_second : ℕ := 6
  let total_kittens : ℕ := blue_eyed_first + brown_eyed_first + blue_eyed_second + brown_eyed_second
  let total_blue_eyed : ℕ := blue_eyed_first + blue_eyed_second
  (total_blue_eyed : ℚ) / total_kittens = 35 / 100 →
  brown_eyed_first = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_cat_brown_eyed_kittens_l3847_384711


namespace NUMINAMATH_CALUDE_cody_caramel_boxes_l3847_384774

-- Define the given conditions
def chocolate_boxes : ℕ := 7
def pieces_per_box : ℕ := 8
def total_pieces : ℕ := 80

-- Define the function to calculate the number of caramel boxes
def caramel_boxes : ℕ :=
  (total_pieces - chocolate_boxes * pieces_per_box) / pieces_per_box

-- Theorem statement
theorem cody_caramel_boxes :
  caramel_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_cody_caramel_boxes_l3847_384774


namespace NUMINAMATH_CALUDE_quadratic_two_members_l3847_384797

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | m * x^2 + 2 * x + 1 = 0}

-- Define the property that A has only two members
def has_two_members (S : Set ℝ) : Prop := ∃ (a b : ℝ), a ≠ b ∧ S = {a, b}

-- Theorem statement
theorem quadratic_two_members :
  ∀ m : ℝ, has_two_members (A m) ↔ (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_members_l3847_384797


namespace NUMINAMATH_CALUDE_grandson_height_prediction_l3847_384743

/-- Predicts the height of the next generation using linear regression -/
def predict_next_height (heights : List ℝ) : ℝ :=
  sorry

theorem grandson_height_prediction 
  (heights : List ℝ) 
  (h1 : heights = [173, 170, 176, 182]) : 
  predict_next_height heights = 185 := by
  sorry

end NUMINAMATH_CALUDE_grandson_height_prediction_l3847_384743


namespace NUMINAMATH_CALUDE_divisor_of_p_l3847_384781

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 75)
  (h4 : 120 < Nat.gcd s p ∧ Nat.gcd s p < 180) :
  5 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_p_l3847_384781


namespace NUMINAMATH_CALUDE_max_value_theorem_l3847_384785

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h1 : a^2 + b^2 - c^2 - d^2 = 0)
  (h2 : a^2 - b^2 - c^2 + d^2 = 56/53 * (b*c + a*d)) :
  (∀ x y z w : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 
    x^2 + y^2 - z^2 - w^2 = 0 ∧
    x^2 - y^2 - z^2 + w^2 = 56/53 * (y*z + x*w) →
    (x*y + z*w) / (y*z + x*w) ≤ (a*b + c*d) / (b*c + a*d)) ∧
  (a*b + c*d) / (b*c + a*d) = 45/53 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3847_384785


namespace NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l3847_384788

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 18 consecutive integers ≤ 2016, one is divisible by its digit sum -/
theorem exists_divisible_by_digit_sum :
  ∀ (start : ℕ), start + 17 ≤ 2016 →
  ∃ n ∈ Finset.range 18, (start + n).mod (sum_of_digits (start + n)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l3847_384788


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l3847_384761

def α : Real := sorry
def n : ℤ := sorry

-- Part 1
theorem trigonometric_ratio_equals_three_fourths 
  (h1 : Real.cos α = -4/5) 
  (h2 : Real.sin α = 3/5) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = 3/4 := by sorry

-- Part 2
theorem trigonometric_expression_equals_negative_four
  (h1 : Real.cos (π + α) = -1/2)
  (h2 : α > 3*π/2 ∧ α < 2*π) :
  (Real.sin (α + (2*n + 1)*π) + Real.sin (α - (2*n + 1)*π)) / 
  (Real.sin (α + 2*n*π) * Real.cos (α - 2*n*π)) = -4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l3847_384761


namespace NUMINAMATH_CALUDE_juans_number_l3847_384791

theorem juans_number (x : ℝ) : ((3 * (x + 3) - 4) / 2 = 10) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_l3847_384791


namespace NUMINAMATH_CALUDE_club_member_selection_l3847_384728

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club -/
def totalMembers : ℕ := 15

/-- The number of members to be chosen -/
def chosenMembers : ℕ := 4

/-- The number of remaining members after excluding the two specific members -/
def remainingMembers : ℕ := totalMembers - 2

theorem club_member_selection :
  choose totalMembers chosenMembers - choose remainingMembers (chosenMembers - 2) = 1287 := by
  sorry

end NUMINAMATH_CALUDE_club_member_selection_l3847_384728


namespace NUMINAMATH_CALUDE_cylinder_height_l3847_384776

/-- The height of a cylinder given its lateral surface area and volume -/
theorem cylinder_height (r h : ℝ) (h1 : 2 * π * r * h = 12 * π) (h2 : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l3847_384776


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l3847_384784

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l3847_384784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3847_384756

/-- An arithmetic sequence with given first term and third term -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 8) :
  a 10 = 29 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3847_384756


namespace NUMINAMATH_CALUDE_log_relationship_l3847_384716

theorem log_relationship (a b : ℝ) : 
  a = Real.log 256 / Real.log 8 → b = Real.log 16 / Real.log 2 → a = (2 * b) / 3 := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l3847_384716


namespace NUMINAMATH_CALUDE_triarc_area_sum_l3847_384737

/-- A region bounded by three circular arcs -/
structure TriarcRegion where
  radius : ℝ
  central_angle : ℝ

/-- The area of a TriarcRegion in the form a√b + cπ -/
structure TriarcArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Compute the area of a TriarcRegion -/
noncomputable def compute_triarc_area (region : TriarcRegion) : TriarcArea :=
  sorry

theorem triarc_area_sum (region : TriarcRegion) 
  (h1 : region.radius = 5)
  (h2 : region.central_angle = 2 * π / 3) : 
  let area := compute_triarc_area region
  area.a + area.b + area.c = -28.25 := by
  sorry

end NUMINAMATH_CALUDE_triarc_area_sum_l3847_384737


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3847_384722

/-- Given two quadratic equations, where the roots of the first are three less than the roots of the second, 
    this theorem proves that the constant term of the first equation is -14.5 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 11*y - 14 = 0 ∧ x = y - 3) →
  c = -14.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3847_384722


namespace NUMINAMATH_CALUDE_breakfast_time_is_39_minutes_l3847_384750

def sausage_count : ℕ := 3
def egg_count : ℕ := 6
def sausage_time : ℕ := 5
def egg_time : ℕ := 4

def total_breakfast_time : ℕ := sausage_count * sausage_time + egg_count * egg_time

theorem breakfast_time_is_39_minutes : total_breakfast_time = 39 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_time_is_39_minutes_l3847_384750


namespace NUMINAMATH_CALUDE_f_shifted_is_even_f_has_three_zeros_l3847_384759

/-- A function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := (x - 1)^2 - |x - 1|

/-- Theorem stating that f(x+1) is an even function on ℝ -/
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 1) = f (-x + 1) := by sorry

/-- Theorem stating that f(x) has exactly three zeros on ℝ -/
theorem f_has_three_zeros : ∃! (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) := by sorry

end NUMINAMATH_CALUDE_f_shifted_is_even_f_has_three_zeros_l3847_384759


namespace NUMINAMATH_CALUDE_number_manipulation_l3847_384717

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l3847_384717


namespace NUMINAMATH_CALUDE_platinum_sphere_weight_in_mercury_l3847_384779

/-- The weight of a platinum sphere in mercury at elevated temperature -/
theorem platinum_sphere_weight_in_mercury
  (p : ℝ)
  (d₁ : ℝ)
  (d₂ : ℝ)
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h_p : p = 30)
  (h_d₁ : d₁ = 21.5)
  (h_d₂ : d₂ = 13.60)
  (h_a₁ : a₁ = 0.0000264)
  (h_a₂ : a₂ = 0.0001815)
  : ∃ w : ℝ, abs (w - 11.310) < 0.001 :=
by
  sorry


end NUMINAMATH_CALUDE_platinum_sphere_weight_in_mercury_l3847_384779


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3847_384715

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_eq2 : a 9 * a 10 = -8) :
  a 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3847_384715


namespace NUMINAMATH_CALUDE_simon_change_calculation_l3847_384707

def pansy_price : ℝ := 2.50
def pansy_quantity : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_quantity : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_quantity : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00

theorem simon_change_calculation :
  let total_before_discount := pansy_price * pansy_quantity + hydrangea_price * hydrangea_quantity + petunia_price * petunia_quantity
  let discount := total_before_discount * discount_rate
  let total_after_discount := total_before_discount - discount
  let change := paid_amount - total_after_discount
  change = 23.00 := by sorry

end NUMINAMATH_CALUDE_simon_change_calculation_l3847_384707


namespace NUMINAMATH_CALUDE_beef_purchase_l3847_384703

theorem beef_purchase (initial_budget : ℕ) (chicken_cost : ℕ) (beef_cost_per_pound : ℕ) (remaining_budget : ℕ)
  (h1 : initial_budget = 80)
  (h2 : chicken_cost = 12)
  (h3 : beef_cost_per_pound = 3)
  (h4 : remaining_budget = 53) :
  (initial_budget - remaining_budget - chicken_cost) / beef_cost_per_pound = 5 := by
  sorry

end NUMINAMATH_CALUDE_beef_purchase_l3847_384703


namespace NUMINAMATH_CALUDE_congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l3847_384778

theorem congruence_existence (A B : ℕ) : Prop :=
  ∃ C : ℕ, C % A = 1 ∧ C % B = 2

theorem no_solution_for_6_8 : ¬(congruence_existence 6 8) := by sorry

theorem solution_exists_for_7_9 : congruence_existence 7 9 := by sorry

end NUMINAMATH_CALUDE_congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l3847_384778


namespace NUMINAMATH_CALUDE_team_reading_balance_l3847_384783

/-- The number of pages in the novel --/
def total_pages : ℕ := 820

/-- Alice's reading speed in seconds per page --/
def alice_speed : ℕ := 25

/-- Bob's reading speed in seconds per page --/
def bob_speed : ℕ := 50

/-- Chandra's reading speed in seconds per page --/
def chandra_speed : ℕ := 35

/-- The number of pages Chandra should read --/
def chandra_pages : ℕ := 482

theorem team_reading_balance :
  bob_speed * (total_pages - chandra_pages) = chandra_speed * chandra_pages := by
  sorry

#check team_reading_balance

end NUMINAMATH_CALUDE_team_reading_balance_l3847_384783


namespace NUMINAMATH_CALUDE_range_of_a_l3847_384795

def f (a x : ℝ) : ℝ := a^2 * x - 2*a + 1

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f a x ≤ 0) → a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3847_384795


namespace NUMINAMATH_CALUDE_inequality_proof_l3847_384766

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + 
  Real.sqrt (c * (1 - a) * (1 - b)) ≤ 1 + Real.sqrt (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3847_384766


namespace NUMINAMATH_CALUDE_vector_sum_example_l3847_384753

theorem vector_sum_example :
  let v1 : Fin 3 → ℝ := ![3, -2, 7]
  let v2 : Fin 3 → ℝ := ![-1, 5, -3]
  v1 + v2 = ![2, 3, 4] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_example_l3847_384753


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3847_384757

theorem least_subtraction_for_divisibility (n m : ℕ) (h : n = 45678 ∧ m = 47) :
  ∃ k : ℕ, k ≤ m - 1 ∧ (n - k) % m = 0 ∧ ∀ j : ℕ, j < k → (n - j) % m ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3847_384757


namespace NUMINAMATH_CALUDE_subset_iff_range_l3847_384734

def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x < x - a}

theorem subset_iff_range (a : ℝ) : A ⊇ B a ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_range_l3847_384734


namespace NUMINAMATH_CALUDE_kramer_packing_theorem_l3847_384792

/-- The number of boxes Kramer can pack per minute -/
def boxes_per_minute : ℕ := 10

/-- The number of cases Kramer can pack in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of minutes in 2 hours -/
def minutes_in_two_hours : ℕ := 2 * 60

/-- The number of boxes of cigarettes in one case -/
def boxes_per_case : ℕ := (boxes_per_minute * minutes_in_two_hours) / cases_in_two_hours

theorem kramer_packing_theorem : boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_kramer_packing_theorem_l3847_384792


namespace NUMINAMATH_CALUDE_palindrome_pairs_exist_l3847_384740

/-- A function that checks if a positive integer is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- A function that generates a palindrome given three digits -/
def generate_palindrome (a b k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are at least 2005 palindrome pairs -/
theorem palindrome_pairs_exist : 
  ∃ (pairs : List (ℕ × ℕ)), pairs.length ≥ 2005 ∧ 
    ∀ (pair : ℕ × ℕ), pair ∈ pairs → 
      is_palindrome pair.1 ∧ is_palindrome pair.2 ∧ pair.2 = pair.1 + 110 :=
sorry

end NUMINAMATH_CALUDE_palindrome_pairs_exist_l3847_384740


namespace NUMINAMATH_CALUDE_remainder_problem_l3847_384725

theorem remainder_problem (x : ℤ) (h : x % 61 = 24) : x % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3847_384725


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3847_384789

theorem smallest_m_satisfying_conditions : ∃ m : ℕ+,
  (∀ k : ℕ+, (∃ n : ℕ, 5 * k = n^5) ∧
             (∃ n : ℕ, 6 * k = n^6) ∧
             (∃ n : ℕ, 7 * k = n^7) →
   m ≤ k) ∧
  (∃ n : ℕ, 5 * m = n^5) ∧
  (∃ n : ℕ, 6 * m = n^6) ∧
  (∃ n : ℕ, 7 * m = n^7) ∧
  m = 2^35 * 3^35 * 5^84 * 7^90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3847_384789


namespace NUMINAMATH_CALUDE_height_ratio_l3847_384772

def sara_height : ℝ := 120 - 82
def joe_height : ℝ := 82

axiom combined_height : sara_height + joe_height = 120
axiom joe_height_relation : ∃ k : ℝ, joe_height = k * sara_height + 6

theorem height_ratio : (joe_height / sara_height) = 41 / 19 := by
  sorry

end NUMINAMATH_CALUDE_height_ratio_l3847_384772


namespace NUMINAMATH_CALUDE_volume_S_form_prism_ratio_l3847_384730

/-- A right rectangular prism with given edge lengths -/
structure RectPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r of a point in the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

theorem volume_S_form (B : RectPrism) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  ∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d :=
sorry

theorem prism_ratio (B : RectPrism) (a b c d : ℝ) :
  B.length = 2 ∧ B.width = 3 ∧ B.height = 5 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (∀ r : ℝ, r ≥ 0 → volume_S B r = a * r^3 + b * r^2 + c * r + d) →
  b * c / (a * d) = 15.5 :=
sorry

end NUMINAMATH_CALUDE_volume_S_form_prism_ratio_l3847_384730


namespace NUMINAMATH_CALUDE_import_tax_threshold_l3847_384767

/-- Proves that the amount in excess of which a 7% import tax was applied is $1,000,
    given that the tax paid was $111.30 on an item with a total value of $2,590. -/
theorem import_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) :
  tax_rate = 0.07 →
  tax_paid = 111.30 →
  total_value = 2590 →
  ∃ (threshold : ℝ), threshold = 1000 ∧ tax_rate * (total_value - threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l3847_384767


namespace NUMINAMATH_CALUDE_vector_dot_product_l3847_384705

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (m, m - 4)
def vector_c (m : ℝ) : ℝ × ℝ := (2*m, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (m : ℝ) :
  parallel vector_a (vector_b m) →
  dot_product (vector_b m) (vector_c m) = -7 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3847_384705


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3847_384721

/-- Given a point P in the second quadrant with absolute x-coordinate 5 and absolute y-coordinate 7,
    the point symmetric to P with respect to the origin has coordinates (5, -7). -/
theorem symmetric_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →  -- Point is in the second quadrant (x is negative)
    y > 0 →  -- Point is in the second quadrant (y is positive)
    |x| = 5 →
    |y| = 7 →
    (- x, - y) = (5, -7) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3847_384721


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3847_384793

/-- Given a geometric sequence with first term 512 and sixth term 32, 
    the fourth term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence property
  a 0 = 512 →                                  -- First term is 512
  a 5 = 32 →                                   -- Sixth term is 32
  a 3 = 64 :=                                  -- Fourth term is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3847_384793


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3847_384732

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3847_384732


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l3847_384782

theorem smallest_multiples_sum (x y : ℕ) : 
  (x ≥ 10 ∧ x < 100 ∧ x % 2 = 0 ∧ ∀ z : ℕ, (z ≥ 10 ∧ z < 100 ∧ z % 2 = 0) → x ≤ z) ∧
  (y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 ∧ ∀ w : ℕ, (w ≥ 100 ∧ w < 1000 ∧ w % 5 = 0) → y ≤ w) →
  2 * (x + y) = 220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l3847_384782


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_1029_l3847_384744

theorem least_n_factorial_divisible_by_1029 : 
  ∃ n : ℕ, n = 21 ∧ 
  (∀ k : ℕ, k < n → ¬(1029 ∣ k!)) ∧ 
  (1029 ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_1029_l3847_384744


namespace NUMINAMATH_CALUDE_square_field_area_l3847_384760

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area (side_length : ℝ) (h : side_length = 20) : 
  side_length * side_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3847_384760


namespace NUMINAMATH_CALUDE_linear_function_problem_l3847_384799

/-- A linear function passing through (1, 3) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- The linear function shifted up by 2 units -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_problem (k : ℝ) (h : k ≠ 0) (h1 : f k 1 = 3) :
  k = 2 ∧ ∀ x, g k x = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l3847_384799


namespace NUMINAMATH_CALUDE_cube_surface_area_difference_l3847_384733

theorem cube_surface_area_difference (large_cube_volume : ℕ) (num_small_cubes : ℕ) (small_cube_volume : ℕ) : 
  large_cube_volume = 6859 →
  num_small_cubes = 6859 →
  small_cube_volume = 1 →
  (num_small_cubes * 6 * small_cube_volume^(2/3) : ℕ) - (6 * large_cube_volume^(2/3) : ℕ) = 38988 := by
  sorry

#eval (6859 * 6 * 1^(2/3) : ℕ) - (6 * 6859^(2/3) : ℕ)

end NUMINAMATH_CALUDE_cube_surface_area_difference_l3847_384733


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l3847_384764

/-- The perimeter of an equilateral triangle, given its relationship with an isosceles triangle -/
theorem equilateral_triangle_perimeter : ℝ :=
  let equilateral_side : ℝ := sorry
  let isosceles_base : ℝ := 10
  let isosceles_perimeter : ℝ := 50
  have h1 : isosceles_perimeter = 2 * equilateral_side + isosceles_base := by sorry
  have h2 : equilateral_side = (isosceles_perimeter - isosceles_base) / 2 := by sorry
  3 * equilateral_side

/-- Proof that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter_is_60 :
  equilateral_triangle_perimeter = 60 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l3847_384764


namespace NUMINAMATH_CALUDE_drug_use_percentage_is_four_percent_l3847_384775

/-- Warner's Random Response Technique for surveying athletes --/
structure WarnerSurvey where
  total_athletes : ℕ
  yes_answers : ℕ
  prob_odd_roll : ℚ
  prob_even_birthday : ℚ

/-- Calculate the percentage of athletes who have used performance-enhancing drugs --/
def calculate_drug_use_percentage (survey : WarnerSurvey) : ℚ :=
  2 * (survey.yes_answers / survey.total_athletes - 1/4)

/-- Theorem stating that the drug use percentage is 4% for the given survey --/
theorem drug_use_percentage_is_four_percent (survey : WarnerSurvey) 
  (h1 : survey.total_athletes = 200)
  (h2 : survey.yes_answers = 54)
  (h3 : survey.prob_odd_roll = 1/2)
  (h4 : survey.prob_even_birthday = 1/2) :
  calculate_drug_use_percentage survey = 4/100 := by
  sorry

end NUMINAMATH_CALUDE_drug_use_percentage_is_four_percent_l3847_384775


namespace NUMINAMATH_CALUDE_expression_value_l3847_384755

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x^2 = 9)  -- distance from x to origin is 3
  : (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3847_384755


namespace NUMINAMATH_CALUDE_stratified_sampling_junior_count_l3847_384770

theorem stratified_sampling_junior_count 
  (total_employees : ℕ) 
  (junior_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : junior_employees = 90) 
  (h3 : sample_size = 30) :
  (junior_employees : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_junior_count_l3847_384770


namespace NUMINAMATH_CALUDE_married_employees_percentage_l3847_384710

-- Define the company
structure Company where
  total_employees : ℕ
  women_percentage : ℚ
  men_single_ratio : ℚ
  women_married_percentage : ℚ

-- Define the conditions
def company_conditions (c : Company) : Prop :=
  c.women_percentage = 64 / 100 ∧
  c.men_single_ratio = 2 / 3 ∧
  c.women_married_percentage = 75 / 100

-- Define the function to calculate the percentage of married employees
def married_percentage (c : Company) : ℚ :=
  let men_percentage := 1 - c.women_percentage
  let married_men := (1 - c.men_single_ratio) * men_percentage
  let married_women := c.women_married_percentage * c.women_percentage
  married_men + married_women

-- Theorem statement
theorem married_employees_percentage (c : Company) :
  company_conditions c → married_percentage c = 60 / 100 := by
  sorry


end NUMINAMATH_CALUDE_married_employees_percentage_l3847_384710


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3847_384747

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - 3*I) = 0 ∧ q 0 = -30) →
  (a = -82/13 ∧ b = 277/13 ∧ c = -390/13) :=
by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3847_384747


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l3847_384773

/-- Given a triangle DEF with side lengths and median, calculate its area using Heron's formula -/
theorem triangle_area_with_median (DE DF DM : ℝ) (h1 : DE = 8) (h2 : DF = 17) (h3 : DM = 11) :
  ∃ (EF : ℝ), let a := DE
               let b := DF
               let c := EF
               let s := (a + b + c) / 2
               (s * (s - a) * (s - b) * (s - c)).sqrt = DM * EF / 2 := by
  sorry

#check triangle_area_with_median

end NUMINAMATH_CALUDE_triangle_area_with_median_l3847_384773


namespace NUMINAMATH_CALUDE_construct_angle_l3847_384714

-- Define the given angle
def given_angle : ℝ := 70

-- Define the target angle
def target_angle : ℝ := 40

-- Theorem statement
theorem construct_angle (straight_angle : ℝ) (right_angle : ℝ) 
  (h1 : straight_angle = 180) 
  (h2 : right_angle = 90) : 
  ∃ (constructed_angle : ℝ), constructed_angle = target_angle :=
sorry

end NUMINAMATH_CALUDE_construct_angle_l3847_384714


namespace NUMINAMATH_CALUDE_mall_sales_problem_l3847_384780

/-- Represents the cost price of the item in yuan -/
def cost_price : ℝ := 500

/-- Represents the markup percentage in the first month -/
def markup1 : ℝ := 0.2

/-- Represents the markup percentage in the second month -/
def markup2 : ℝ := 0.1

/-- Represents the profit in the first month in yuan -/
def profit1 : ℝ := 6000

/-- Represents the increase in profit in the second month in yuan -/
def profit_increase : ℝ := 2000

/-- Represents the increase in sales volume in the second month -/
def sales_increase : ℕ := 100

/-- Theorem stating the cost price and second month sales volume -/
theorem mall_sales_problem :
  (cost_price * markup1 * (profit1 / (cost_price * markup1)) +
   cost_price * markup2 * ((profit1 + profit_increase) / (cost_price * markup2)) -
   cost_price * markup1 * (profit1 / (cost_price * markup1))) / cost_price = sales_increase ∧
  (profit1 + profit_increase) / (cost_price * markup2) = 160 :=
by sorry

end NUMINAMATH_CALUDE_mall_sales_problem_l3847_384780


namespace NUMINAMATH_CALUDE_fraction_proof_l3847_384763

theorem fraction_proof (n : ℝ) (f : ℝ) (h1 : n / 2 = 945.0000000000013) 
  (h2 : (4/15 * 5/7 * n) - (4/9 * f * n) = 24) : f = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l3847_384763


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3847_384751

theorem smallest_positive_integer_with_remainders : ∃ M : ℕ+,
  (M : ℕ) % 3 = 2 ∧
  (M : ℕ) % 4 = 3 ∧
  (M : ℕ) % 5 = 4 ∧
  (M : ℕ) % 6 = 5 ∧
  (M : ℕ) % 7 = 6 ∧
  (∀ n : ℕ+, n < M →
    (n : ℕ) % 3 ≠ 2 ∨
    (n : ℕ) % 4 ≠ 3 ∨
    (n : ℕ) % 5 ≠ 4 ∨
    (n : ℕ) % 6 ≠ 5 ∨
    (n : ℕ) % 7 ≠ 6) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3847_384751


namespace NUMINAMATH_CALUDE_expression_evaluation_l3847_384709

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3847_384709


namespace NUMINAMATH_CALUDE_mean_median_difference_l3847_384704

-- Define the score distribution
def score_60_percent : ℝ := 0.20
def score_75_percent : ℝ := 0.40
def score_85_percent : ℝ := 0.25
def score_95_percent : ℝ := 1 - (score_60_percent + score_75_percent + score_85_percent)

-- Define the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

-- Calculate the mean score
def mean_score : ℝ :=
  score_60_percent * score_60 +
  score_75_percent * score_75 +
  score_85_percent * score_85 +
  score_95_percent * score_95

-- Define the median score
def median_score : ℝ := score_75

-- Theorem stating the difference between mean and median
theorem mean_median_difference :
  |mean_score - median_score| = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3847_384704


namespace NUMINAMATH_CALUDE_box_fill_rate_l3847_384723

-- Define the box dimensions
def box_length : ℝ := 7
def box_width : ℝ := 6
def box_height : ℝ := 2

-- Define the time to fill the box
def fill_time : ℝ := 21

-- Calculate the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- Define the theorem
theorem box_fill_rate :
  box_volume / fill_time = 4 := by sorry

end NUMINAMATH_CALUDE_box_fill_rate_l3847_384723


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3847_384798

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 80) 
  (h2 : math = 50) 
  (h3 : physics = 40) 
  (h4 : both = 25) : 
  total - (math + physics - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3847_384798


namespace NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l3847_384729

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) (h1 : speed1 = 4.2)
    (h2 : speed2 = 3.8) (h3 : meeting_time = 4.8 / 60) : ℝ :=
  let distance1 := speed1 * meeting_time
  let distance2 := speed2 * meeting_time
  let total_distance := distance1 + distance2
  total_distance

/-- The circumference of the jogging track is 0.63984 km -/
theorem jogging_track_circumference :
    track_circumference 4.2 3.8 (4.8 / 60) rfl rfl rfl = 0.63984 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l3847_384729


namespace NUMINAMATH_CALUDE_total_ears_is_500_l3847_384790

/-- Calculates the total number of ears for a given number of puppies -/
def total_ears (total_puppies droopy_eared_puppies pointed_eared_puppies : ℕ) : ℕ :=
  2 * total_puppies

/-- Theorem stating that the total number of ears is 500 given the problem conditions -/
theorem total_ears_is_500 :
  let total_puppies : ℕ := 250
  let droopy_eared_puppies : ℕ := 150
  let pointed_eared_puppies : ℕ := 100
  total_ears total_puppies droopy_eared_puppies pointed_eared_puppies = 500 := by
  sorry


end NUMINAMATH_CALUDE_total_ears_is_500_l3847_384790


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3847_384739

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3847_384739


namespace NUMINAMATH_CALUDE_at_least_two_babies_speak_l3847_384758

def probability_baby_speaks : ℚ := 1 / 5

def number_of_babies : ℕ := 7

theorem at_least_two_babies_speak :
  let p := probability_baby_speaks
  let n := number_of_babies
  (1 : ℚ) - (1 - p)^n - n * p * (1 - p)^(n-1) = 50477 / 78125 :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_babies_speak_l3847_384758


namespace NUMINAMATH_CALUDE_brand_preference_l3847_384771

theorem brand_preference (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 180 →
  ratio = 5 / 1 →
  brand_x * (1 + 1 / ratio) = total →
  brand_x = 150 :=
by sorry

end NUMINAMATH_CALUDE_brand_preference_l3847_384771


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3847_384724

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3847_384724


namespace NUMINAMATH_CALUDE_student_weight_is_90_l3847_384736

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The combined weight of the student and his sister in kilograms -/
def combined_weight : ℝ := 132

/-- If the student loses 6 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 6 = 2 * sister_weight

/-- The combined weight of the student and his sister is 132 kilograms -/
axiom total_weight : student_weight + sister_weight = combined_weight

/-- Theorem: The student's present weight is 90 kilograms -/
theorem student_weight_is_90 : student_weight = 90 := by sorry

end NUMINAMATH_CALUDE_student_weight_is_90_l3847_384736


namespace NUMINAMATH_CALUDE_friday_fries_ratio_l3847_384727

/-- Represents the number of fries sold -/
structure FriesSold where
  total : ℕ
  small : ℕ

/-- Calculates the ratio of large fries to small fries -/
def largeToSmallRatio (fs : FriesSold) : ℚ :=
  (fs.total - fs.small : ℚ) / fs.small

theorem friday_fries_ratio :
  let fs : FriesSold := { total := 24, small := 4 }
  largeToSmallRatio fs = 5 := by
  sorry

end NUMINAMATH_CALUDE_friday_fries_ratio_l3847_384727


namespace NUMINAMATH_CALUDE_regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l3847_384746

/-- A regular polygon -/
structure RegularPolygon where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The radius of the circumscribed circle -/
  circumRadius : ℝ
  /-- Assertion that the number of sides is at least 3 -/
  sidesGe3 : sides ≥ 3

/-- Theorem stating that a regular polygon is not uniquely determined by its circumradius -/
theorem regular_polygon_not_unique_by_circumradius :
  ∃ (p q : RegularPolygon), p.circumRadius = q.circumRadius ∧ p.sides ≠ q.sides :=
sorry

/-- Corollary specifically for triangles -/
theorem triangle_not_unique_by_circumradius :
  ∃ (t : RegularPolygon) (p : RegularPolygon), 
    t.sides = 3 ∧ p.sides ≠ 3 ∧ t.circumRadius = p.circumRadius :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l3847_384746


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3847_384769

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 3 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l3847_384769


namespace NUMINAMATH_CALUDE_painter_problem_solution_l3847_384706

/-- Given a painting job with a total number of rooms, time per room, and some rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the specific problem, the time to paint the remaining rooms is 63 hours. -/
theorem painter_problem_solution :
  time_to_paint_remaining 11 7 2 = 63 := by
  sorry

#eval time_to_paint_remaining 11 7 2

end NUMINAMATH_CALUDE_painter_problem_solution_l3847_384706


namespace NUMINAMATH_CALUDE_min_value_of_a_l3847_384745

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / (Real.exp x)

theorem min_value_of_a (a : ℝ) :
  (∃ x : ℝ, x ≥ -2 ∧ f x ≤ a) ↔ a ≥ 1 - 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3847_384745


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3847_384748

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3847_384748


namespace NUMINAMATH_CALUDE_max_shadow_distance_l3847_384742

/-- 
Given a projectile motion with:
- v: initial velocity
- t: time of flight
- y: vertical displacement
- g: gravitational acceleration
- a: constant horizontal acceleration due to air resistance

The maximum horizontal distance L of the projectile's shadow is 0.75 m.
-/
theorem max_shadow_distance 
  (v : ℝ) 
  (t : ℝ) 
  (y : ℝ) 
  (g : ℝ) 
  (a : ℝ) 
  (h1 : v = 5)
  (h2 : t = 1)
  (h3 : y = -1)
  (h4 : g = 10)
  (h5 : y = v * Real.sin α * t - (g * t^2) / 2)
  (h6 : 0 = v * Real.cos α * t - (a * t^2) / 2)
  (h7 : α = Real.arcsin (4/5))
  : ∃ L : ℝ, L = 0.75 ∧ L = (v^2 * (Real.cos α)^2) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_max_shadow_distance_l3847_384742


namespace NUMINAMATH_CALUDE_john_total_skateboard_distance_l3847_384720

/-- The total distance John skateboarded, given his trip to and from the park -/
def total_skateboard_distance (distance_to_park : ℕ) : ℕ :=
  2 * distance_to_park

/-- Theorem: John skateboarded a total of 32 miles -/
theorem john_total_skateboard_distance :
  total_skateboard_distance 16 = 32 :=
by sorry

end NUMINAMATH_CALUDE_john_total_skateboard_distance_l3847_384720


namespace NUMINAMATH_CALUDE_inverse_proportion_l3847_384700

theorem inverse_proportion (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x is inversely proportional to y
  x 2 = 4 →               -- x = 4 when y = 2
  y 2 = 2 →               -- y = 2 when x = 4
  y (-5) = -5 →           -- y = -5
  x (-5) = -8/5 :=        -- x = -8/5 when y = -5
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_l3847_384700


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3847_384712

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 →
  cone_volume + hemisphere_volume = 48 * π := by
sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3847_384712


namespace NUMINAMATH_CALUDE_ellipse_theorems_l3847_384762

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and focal length 2√3, prove the following theorems. -/
theorem ellipse_theorems 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_focal : a^2 - b^2 = 3) :
  let C : ℝ × ℝ → Prop := λ p => p.1^2 / 4 + p.2^2 = 1
  ∃ (k : ℝ) (h_k : k ≠ 0),
    let l₁ : ℝ → ℝ := λ x => k * x
    ∃ (A B : ℝ × ℝ) (h_AB : A.2 = l₁ A.1 ∧ B.2 = l₁ B.1),
      let l₂ : ℝ → ℝ := λ x => (B.2 + k/4 * (x - B.1))
      ∃ (D : ℝ × ℝ) (h_D : D.2 = l₂ D.1),
        (A.1 - D.1) * (A.1 - B.1) + (A.2 - D.2) * (A.2 - B.2) = 0 →
        (∀ p : ℝ × ℝ, C p ↔ p.1^2 / 4 + p.2^2 = 1) ∧
        (∃ (M N : ℝ × ℝ), 
          M.2 = 0 ∧ N.1 = 0 ∧ M.2 = l₂ M.1 ∧ N.2 = l₂ N.1 ∧
          ∀ (M' N' : ℝ × ℝ), M'.2 = 0 ∧ N'.1 = 0 ∧ M'.2 = l₂ M'.1 ∧ N'.2 = l₂ N'.1 →
          abs (M.1 * N.2) / 2 ≥ abs (M'.1 * N'.2) / 2 ∧
          abs (M.1 * N.2) / 2 = 9/8) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_theorems_l3847_384762


namespace NUMINAMATH_CALUDE_common_divisors_9240_10800_l3847_384738

theorem common_divisors_9240_10800 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10800} = 16 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10800_l3847_384738


namespace NUMINAMATH_CALUDE_equality_of_cyclic_sum_powers_l3847_384794

theorem equality_of_cyclic_sum_powers (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_cycle : a^n.val + p * b = b^n.val + p * c ∧ b^n.val + p * c = c^n.val + p * a) :
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_sum_powers_l3847_384794


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l3847_384741

/-- The distance between two points A and B, where two people walk towards each other
    under specific conditions. -/
def distance_AB : ℝ := 2800

theorem meeting_point_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := distance_AB
  let meeting_point := 1200
  let B_double_speed := 2 * y
  S / 2 / x + (S / 2 - meeting_point) / x = 
    S / 2 / y + (meeting_point - S * y / (2 * x)) / B_double_speed ∧
  S - meeting_point = S / 2 →
  S = 2800 := by sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l3847_384741


namespace NUMINAMATH_CALUDE_teairra_closet_count_l3847_384787

/-- The number of shirts and pants that are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

theorem teairra_closet_count :
  non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_teairra_closet_count_l3847_384787


namespace NUMINAMATH_CALUDE_min_value_m_min_value_m_tight_l3847_384719

theorem min_value_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) → m ≥ 2 * Real.sqrt 3 :=
by sorry

theorem min_value_m_tight : 
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) ∧ m = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_min_value_m_tight_l3847_384719


namespace NUMINAMATH_CALUDE_light_bulb_replacement_l3847_384786

def month_number (m : String) : Nat :=
  match m with
  | "January" => 1
  | "February" => 2
  | "March" => 3
  | "April" => 4
  | "May" => 5
  | "June" => 6
  | "July" => 7
  | "August" => 8
  | "September" => 9
  | "October" => 10
  | "November" => 11
  | "December" => 12
  | _ => 0

def cycle_length : Nat := 7
def start_month : String := "January"
def replacement_count : Nat := 12

theorem light_bulb_replacement :
  (cycle_length * (replacement_count - 1)) % 12 + month_number start_month = month_number "June" :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_replacement_l3847_384786


namespace NUMINAMATH_CALUDE_solution_comparison_l3847_384726

theorem solution_comparison (c c' d d' : ℝ) (hc : c ≠ 0) (hc' : c' ≠ 0) :
  (-d / c > -d' / c') ↔ (d' / c' < d / c) := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l3847_384726


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l3847_384731

-- (1) Prove that x = 3 or x = -1 is a solution to 4(x-1)^2 - 16 = 0
theorem problem_1 : ∃ x : ℝ, (x = 3 ∨ x = -1) ∧ 4 * (x - 1)^2 - 16 = 0 := by sorry

-- (2) Prove that ∛(-64) + √16 * √(9/4) + (-√2)^2 = 4
theorem problem_2 : ((-64 : ℝ)^(1/3)) + Real.sqrt 16 * Real.sqrt (9/4) + (-Real.sqrt 2)^2 = 4 := by sorry

-- (3) Prove that if a is the integer part and b is the decimal part of 9 - √13, then 2a + b = 14 - √13
theorem problem_3 (a b : ℝ) (h : a = ⌊9 - Real.sqrt 13⌋ ∧ b = 9 - Real.sqrt 13 - a) :
  2 * a + b = 14 - Real.sqrt 13 := by sorry

-- (4) Define an operation ⊕ and prove that x = 5 or x = -5 is a solution to (4 ⊕ 3) ⊕ x = 24
def circle_plus (a b : ℝ) : ℝ := a^2 - b^2

theorem problem_4 : ∃ x : ℝ, (x = 5 ∨ x = -5) ∧ circle_plus (circle_plus 4 3) x = 24 := by sorry

-- (5) Prove that if ∠1 and ∠2 are parallel, and ∠1 is 36° less than three times ∠2, then ∠1 = 18° or ∠1 = 126°
theorem problem_5 (angle1 angle2 : ℝ) 
  (h1 : angle1 = 3 * angle2 - 36)
  (h2 : angle1 = angle2 ∨ angle1 + angle2 = 180) :
  angle1 = 18 ∨ angle1 = 126 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l3847_384731
