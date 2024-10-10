import Mathlib

namespace halloween_candy_proof_l3076_307644

/-- The number of candy pieces Faye scored on Halloween -/
def initial_candy : ℕ := 47

/-- The number of candy pieces Faye ate on the first night -/
def eaten_candy : ℕ := 25

/-- The number of candy pieces Faye's sister gave her -/
def gifted_candy : ℕ := 40

/-- The number of candy pieces Faye has now -/
def current_candy : ℕ := 62

/-- Theorem stating that the initial number of candy pieces is correct -/
theorem halloween_candy_proof : 
  initial_candy - eaten_candy + gifted_candy = current_candy :=
by sorry

end halloween_candy_proof_l3076_307644


namespace elizabeth_granola_profit_l3076_307601

/-- Calculate Elizabeth's net profit from selling granola bags --/
theorem elizabeth_granola_profit :
  let ingredient_cost_per_bag : ℚ := 3
  let total_bags : ℕ := 20
  let full_price : ℚ := 6
  let full_price_sales : ℕ := 15
  let discounted_price : ℚ := 4
  let discounted_sales : ℕ := 5

  let total_cost : ℚ := ingredient_cost_per_bag * total_bags
  let full_price_revenue : ℚ := full_price * full_price_sales
  let discounted_revenue : ℚ := discounted_price * discounted_sales
  let total_revenue : ℚ := full_price_revenue + discounted_revenue
  let net_profit : ℚ := total_revenue - total_cost

  net_profit = 50 := by sorry

end elizabeth_granola_profit_l3076_307601


namespace bruce_fruit_purchase_l3076_307619

/-- Calculates the total amount paid for fruits given their quantities and rates. -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 1055 to the shopkeeper for his fruit purchase. -/
theorem bruce_fruit_purchase : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end bruce_fruit_purchase_l3076_307619


namespace system_solution_unique_l3076_307695

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2^(x + y) = x + 7) ∧ (x + y = 3) :=
by sorry

end system_solution_unique_l3076_307695


namespace ada_paul_test_scores_l3076_307647

/-- Ada and Paul's test scores problem -/
theorem ada_paul_test_scores 
  (a1 a2 a3 p1 p2 p3 : ℤ) 
  (h1 : a1 = p1 + 10)
  (h2 : a2 = p2 + 4)
  (h3 : (p1 + p2 + p3) / 3 = (a1 + a2 + a3) / 3 + 4) :
  p3 - a3 = 26 := by
sorry

end ada_paul_test_scores_l3076_307647


namespace zeros_after_decimal_point_of_inverse_40_power_20_l3076_307645

theorem zeros_after_decimal_point_of_inverse_40_power_20 :
  let n : ℕ := 40
  let p : ℕ := 20
  let f : ℚ := 1 / (n^p : ℚ)
  (∃ (x : ℚ) (k : ℕ), f = x * 10^(-k : ℤ) ∧ x ≥ 1/10 ∧ x < 1 ∧ k = 38) :=
by sorry

end zeros_after_decimal_point_of_inverse_40_power_20_l3076_307645


namespace distance_to_origin_l3076_307699

def A : ℝ × ℝ := (-1, -2)

theorem distance_to_origin : Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = Real.sqrt 5 := by
  sorry

end distance_to_origin_l3076_307699


namespace least_valid_n_l3076_307664

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n + 1 ∧ 1 ≤ k₂ ∧ k₂ ≤ n + 1 ∧
  (n^2 - n) % k₁ = 0 ∧ (n^2 - n) % k₂ ≠ 0

theorem least_valid_n :
  is_valid 5 ∧ ∀ m : ℕ, m < 5 → ¬is_valid m :=
sorry

end least_valid_n_l3076_307664


namespace water_percentage_in_fresh_grapes_l3076_307652

/-- Given that dried grapes contain 20% water by weight and 40 kg of fresh grapes
    produce 5 kg of dried grapes, prove that the percentage of water in fresh grapes is 90%. -/
theorem water_percentage_in_fresh_grapes :
  ∀ (fresh_weight dried_weight : ℝ) (dried_water_percentage : ℝ),
    fresh_weight = 40 →
    dried_weight = 5 →
    dried_water_percentage = 20 →
    (fresh_weight - dried_weight * (1 - dried_water_percentage / 100)) / fresh_weight * 100 = 90 := by
  sorry

end water_percentage_in_fresh_grapes_l3076_307652


namespace cubic_and_quadratic_equations_l3076_307600

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, 8 * x^3 = 27 ∧ x = 3/2) ∧
  (∃ x y : ℝ, (x - 2)^2 = 3 ∧ (y - 2)^2 = 3 ∧ 
   x = Real.sqrt 3 + 2 ∧ y = -Real.sqrt 3 + 2) :=
by sorry

end cubic_and_quadratic_equations_l3076_307600


namespace commission_calculation_l3076_307639

/-- The commission percentage for sales exceeding $500 -/
def excess_commission_percentage : ℝ :=
  -- We'll define this value and prove it's equal to 50
  50

theorem commission_calculation (total_sale : ℝ) (total_commission_percentage : ℝ) :
  total_sale = 800 →
  total_commission_percentage = 31.25 →
  (0.2 * 500 + (excess_commission_percentage / 100) * (total_sale - 500)) / total_sale = total_commission_percentage / 100 →
  excess_commission_percentage = 50 := by
  sorry

#check commission_calculation

end commission_calculation_l3076_307639


namespace no_three_reals_satisfying_conditions_l3076_307653

/-- Definition of the set S(a) -/
def S (a : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * a⌋}

/-- Theorem stating the impossibility of finding three positive reals satisfying the given conditions -/
theorem no_three_reals_satisfying_conditions :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧
    (S a ∪ S b ∪ S c = Set.univ) :=
sorry

end no_three_reals_satisfying_conditions_l3076_307653


namespace unique_solution_abc_l3076_307648

theorem unique_solution_abc :
  ∀ A B C : ℝ,
  A = 2 * B - 3 * C →
  B = 2 * C - 5 →
  A + B + C = 100 →
  A = 18.75 ∧ B = 52.5 ∧ C = 28.75 := by
sorry

end unique_solution_abc_l3076_307648


namespace soccer_camp_afternoon_l3076_307634

/-- Given 2000 kids in total, with half going to soccer camp and 1/4 of those going in the morning,
    the number of kids going to soccer camp in the afternoon is 750. -/
theorem soccer_camp_afternoon (total : ℕ) (soccer : ℕ) (morning : ℕ) : 
  total = 2000 →
  soccer = total / 2 →
  morning = soccer / 4 →
  soccer - morning = 750 := by
  sorry

end soccer_camp_afternoon_l3076_307634


namespace juan_running_l3076_307694

/-- The distance traveled when moving at a constant speed for a given time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Juan's running problem -/
theorem juan_running :
  let speed : ℝ := 10  -- miles per hour
  let time : ℝ := 8    -- hours
  distance speed time = 80 := by
  sorry

end juan_running_l3076_307694


namespace y_exceeds_x_l3076_307698

theorem y_exceeds_x (x y : ℝ) (h : x = 0.75 * y) : (y - x) / x = 1/3 := by
  sorry

end y_exceeds_x_l3076_307698


namespace textbook_order_cost_l3076_307612

/-- Calculate the total cost of textbooks --/
def total_cost (english_count : ℕ) (english_price : ℚ)
                (geography_count : ℕ) (geography_price : ℚ)
                (math_count : ℕ) (math_price : ℚ)
                (science_count : ℕ) (science_price : ℚ) : ℚ :=
  english_count * english_price +
  geography_count * geography_price +
  math_count * math_price +
  science_count * science_price

/-- The total cost of the textbook order is $1155.00 --/
theorem textbook_order_cost :
  total_cost 35 (7.5) 35 (10.5) 20 12 30 (9.5) = 1155 := by
  sorry

end textbook_order_cost_l3076_307612


namespace river_width_is_500_l3076_307675

/-- Represents the river crossing scenario -/
structure RiverCrossing where
  velocity : ℝ  -- Boatman's velocity in m/sec
  time : ℝ      -- Time taken to cross the river in seconds
  drift : ℝ     -- Drift distance in meters

/-- Calculates the width of the river given the crossing parameters -/
def riverWidth (rc : RiverCrossing) : ℝ :=
  rc.velocity * rc.time

/-- Theorem stating that the width of the river is 500 meters 
    given the specific conditions -/
theorem river_width_is_500 (rc : RiverCrossing) 
  (h1 : rc.velocity = 10)
  (h2 : rc.time = 50)
  (h3 : rc.drift = 300) : 
  riverWidth rc = 500 := by
  sorry

#check river_width_is_500

end river_width_is_500_l3076_307675


namespace geometric_sequence_common_ratio_l3076_307651

/-- A geometric sequence with first term a₁ = -1 and a₂ + a₃ = -2 has common ratio q = -2 or q = 1 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  q = -2 ∨ q = 1 :=
sorry

end geometric_sequence_common_ratio_l3076_307651


namespace triangle_trig_inequality_triangle_trig_equality_l3076_307613

/-- For any triangle ABC, sin A + sin B sin C + cos B cos C ≤ 2 -/
theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B * Real.sin C + Real.cos B * Real.cos C ≤ 2 :=
sorry

/-- The equality holds when A = π/2 and B = C = π/4 -/
theorem triangle_trig_equality :
  Real.sin (Real.pi/2) + Real.sin (Real.pi/4) * Real.sin (Real.pi/4) + 
  Real.cos (Real.pi/4) * Real.cos (Real.pi/4) = 2 :=
sorry

end triangle_trig_inequality_triangle_trig_equality_l3076_307613


namespace marble_distribution_marble_distribution_proof_l3076_307658

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joined : ℕ) : Prop :=
  total_marbles = 180 →
  initial_group = 18 →
  (total_marbles / initial_group : ℚ) - (total_marbles / (initial_group + joined) : ℚ) = 1 →
  joined = 2

-- The proof would go here, but we'll use sorry as requested
theorem marble_distribution_proof : marble_distribution 180 18 2 := by sorry

end marble_distribution_marble_distribution_proof_l3076_307658


namespace divisiblity_by_thirty_l3076_307689

theorem divisiblity_by_thirty (p : ℕ) (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) :
  ∃ k : ℕ, p^2 - 1 = 30 * k := by
  sorry

end divisiblity_by_thirty_l3076_307689


namespace heartsuit_three_eight_l3076_307642

-- Define the ♡ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_three_eight_l3076_307642


namespace original_number_reciprocal_l3076_307614

theorem original_number_reciprocal (x : ℝ) : 1 / x - 3 = 5 / 2 → x = 2 / 11 := by
  sorry

end original_number_reciprocal_l3076_307614


namespace quadratic_equation_rewrite_l3076_307659

theorem quadratic_equation_rewrite (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
  sorry

end quadratic_equation_rewrite_l3076_307659


namespace three_X_seven_equals_eight_l3076_307625

/-- The operation X defined for two real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - a^2 - 5 * b

/-- Theorem stating that 3X7 equals 8 -/
theorem three_X_seven_equals_eight : X 3 7 = 8 := by
  sorry

end three_X_seven_equals_eight_l3076_307625


namespace percentage_rejected_l3076_307669

theorem percentage_rejected (john_rejection_rate jane_rejection_rate jane_inspection_fraction : ℝ) 
  (h1 : john_rejection_rate = 0.005)
  (h2 : jane_rejection_rate = 0.008)
  (h3 : jane_inspection_fraction = 0.8333333333333333)
  : jane_rejection_rate * jane_inspection_fraction + 
    john_rejection_rate * (1 - jane_inspection_fraction) = 0.0075 := by
  sorry

end percentage_rejected_l3076_307669


namespace pure_imaginary_value_l3076_307681

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_value (x : ℝ) :
  is_pure_imaginary ((x^2 - 1 : ℝ) + (x^2 + 3*x + 2 : ℝ) * I) → x = 1 := by
  sorry

end pure_imaginary_value_l3076_307681


namespace cos_three_pi_four_plus_two_alpha_l3076_307641

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end cos_three_pi_four_plus_two_alpha_l3076_307641


namespace committee_meeting_arrangements_l3076_307630

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club with its schools --/
structure Club :=
  (schools : List School)
  (total_members : Nat)

/-- Represents the committee meeting arrangement --/
structure CommitteeMeeting :=
  (host : School)
  (first_non_host : School)
  (second_non_host : School)
  (host_reps : Nat)
  (first_non_host_reps : Nat)
  (second_non_host_reps : Nat)

/-- The number of ways to arrange a committee meeting --/
def arrange_committee_meeting (club : Club) : Nat :=
  sorry

/-- Theorem stating the number of possible committee meeting arrangements --/
theorem committee_meeting_arrangements (club : Club) :
  club.schools.length = 3 ∧
  club.total_members = 18 ∧
  (∀ s ∈ club.schools, s.members = 6) →
  arrange_committee_meeting club = 5400 :=
sorry

end committee_meeting_arrangements_l3076_307630


namespace quadratic_rewrite_l3076_307672

theorem quadratic_rewrite (j : ℝ) : 
  ∃ (c p q : ℝ), 9 * j^2 - 12 * j + 27 = c * (j + p)^2 + q ∧ q / p = -69 / 2 := by
sorry

end quadratic_rewrite_l3076_307672


namespace distance_between_centers_l3076_307690

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 80^2
  xz_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 150^2
  yz_length : (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 170^2

-- Define the inscribed circle C₁
def InscribedCircle (T : Triangle X Y Z) (C : ℝ × ℝ) (r : ℝ) : Prop := sorry

-- Define MN perpendicular to XZ and tangent to C₁
def MN_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (M N : ℝ × ℝ) : Prop := sorry

-- Define AB perpendicular to XY and tangent to C₁
def AB_Perpendicular_Tangent (T : Triangle X Y Z) (C₁ : ℝ × ℝ) (r₁ : ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define the inscribed circle C₂ of MZN
def InscribedCircle_MZN (T : Triangle X Y Z) (M N : ℝ × ℝ) (C₂ : ℝ × ℝ) (r₂ : ℝ) : Prop := sorry

-- Define the inscribed circle C₃ of YAB
def InscribedCircle_YAB (T : Triangle X Y Z) (A B : ℝ × ℝ) (C₃ : ℝ × ℝ) (r₃ : ℝ) : Prop := sorry

theorem distance_between_centers (X Y Z M N A B C₁ C₂ C₃ : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) 
  (h_triangle : Triangle X Y Z)
  (h_c₁ : InscribedCircle h_triangle C₁ r₁)
  (h_mn : MN_Perpendicular_Tangent h_triangle C₁ r₁ M N)
  (h_ab : AB_Perpendicular_Tangent h_triangle C₁ r₁ A B)
  (h_c₂ : InscribedCircle_MZN h_triangle M N C₂ r₂)
  (h_c₃ : InscribedCircle_YAB h_triangle A B C₃ r₃) :
  (C₂.1 - C₃.1)^2 + (C₂.2 - C₃.2)^2 = 9884.5 := by sorry

end distance_between_centers_l3076_307690


namespace imaginary_part_of_complex_fraction_l3076_307687

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I^3 / (1 - Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l3076_307687


namespace circle_equation_l3076_307678

/-- Theorem: Given a circle with center (a, 0) where a < 0, radius √5, and tangent to the line x + 2y = 0, 
    the equation of the circle is (x + 5)² + y² = 5. -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let r : ℝ := Real.sqrt 5
  let d : ℝ → ℝ → ℝ := λ x y => |x + 2*y| / Real.sqrt 5
  (d a 0 = r) → 
  (∀ x y, (x - a)^2 + y^2 = 5 ↔ (x + 5)^2 + y^2 = 5) :=
by sorry

end circle_equation_l3076_307678


namespace triangle_properties_l3076_307609

/-- Given a triangle ABC with dot product conditions, prove the length of AB and a trigonometric ratio -/
theorem triangle_properties (A B C : ℝ × ℝ) 
  (h1 : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 9)
  (h2 : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = -16) :
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let cosA := ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (AB * CA)
  let cosB := ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / (AB * BC)
  let cosC := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (BC * CA)
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := Real.sqrt (1 - cosC^2)
  AB = 5 ∧ (sinA * cosB - cosA * sinB) / sinC = 7/25 := by
  sorry

end triangle_properties_l3076_307609


namespace pumpkin_relationship_other_orchard_pumpkins_l3076_307628

/-- Represents the number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 54

/-- Represents the number of pumpkins at the other orchard -/
def other_pumpkins : ℕ := 14

/-- Theorem stating the relationship between the number of pumpkins at Sunshine Orchard and the other orchard -/
theorem pumpkin_relationship : sunshine_pumpkins = 3 * other_pumpkins + 12 := by
  sorry

/-- Theorem proving that the other orchard has 14 pumpkins given the conditions -/
theorem other_orchard_pumpkins : other_pumpkins = 14 := by
  sorry

end pumpkin_relationship_other_orchard_pumpkins_l3076_307628


namespace puzzle_solution_l3076_307635

theorem puzzle_solution :
  ∃ (g n o u w : Nat),
    g ∈ Finset.range 10 ∧
    n ∈ Finset.range 10 ∧
    o ∈ Finset.range 10 ∧
    u ∈ Finset.range 10 ∧
    w ∈ Finset.range 10 ∧
    (100 * g + 10 * u + n) ^ 2 = 100000 * w + 10000 * o + 1000 * w + 100 * g + 10 * u + n ∧
    o - w = 3 :=
by sorry

end puzzle_solution_l3076_307635


namespace angle_value_proof_l3076_307661

/-- Given that cos 16° = sin 14° + sin d° and 0 < d < 90, prove that d = 46 -/
theorem angle_value_proof (d : ℝ) 
  (h1 : Real.cos (16 * π / 180) = Real.sin (14 * π / 180) + Real.sin (d * π / 180))
  (h2 : 0 < d)
  (h3 : d < 90) : 
  d = 46 := by
  sorry

end angle_value_proof_l3076_307661


namespace derivative_problems_l3076_307682

open Real

theorem derivative_problems :
  (∀ x > 0, deriv (λ x => (log x) / x) x = (1 - log x) / x^2) ∧
  (∀ x, deriv (λ x => x * exp x) x = (x + 1) * exp x) ∧
  (∀ x, deriv (λ x => cos (2 * x)) x = -2 * sin (2 * x)) :=
by sorry

end derivative_problems_l3076_307682


namespace triangle_problem_l3076_307623

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  A = 30 * π / 180 ∧  -- Convert 30° to radians
  a = 2 ∧
  b = 2 * Real.sqrt 3 ∧
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧
  ∃! (B' C' : Real), B' ≠ B ∧ 
    A + B + C = π ∧
    A + B' + C' = π ∧
    a / Real.sin A = b / Real.sin B' ∧
    b / Real.sin B' = c / Real.sin C' ∧
    c / Real.sin C' = a / Real.sin A :=
by sorry


end triangle_problem_l3076_307623


namespace min_value_of_function_l3076_307684

theorem min_value_of_function (x : ℝ) (h : x > 4) : 
  ∃ (y_min : ℝ), y_min = 6 ∧ ∀ y, y = x + 1 / (x - 4) → y ≥ y_min :=
by sorry

end min_value_of_function_l3076_307684


namespace max_value_of_5x_minus_25x_l3076_307618

theorem max_value_of_5x_minus_25x : 
  ∃ (max : ℝ), max = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ max :=
sorry

end max_value_of_5x_minus_25x_l3076_307618


namespace number_multiplying_a_l3076_307691

theorem number_multiplying_a (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a / 4 = b / 3) :
  ∃ x : ℝ, x * a = 4 * b ∧ x = 3 := by
  sorry

end number_multiplying_a_l3076_307691


namespace sally_cost_is_42000_l3076_307665

/-- The cost of Lightning McQueen in dollars -/
def lightning_cost : ℝ := 140000

/-- The cost of Mater as a percentage of Lightning McQueen's cost -/
def mater_percentage : ℝ := 0.10

/-- The factor by which Sally McQueen's cost is greater than Mater's cost -/
def sally_factor : ℝ := 3

/-- The cost of Sally McQueen in dollars -/
def sally_cost : ℝ := lightning_cost * mater_percentage * sally_factor

theorem sally_cost_is_42000 : sally_cost = 42000 := by
  sorry

end sally_cost_is_42000_l3076_307665


namespace proportion_fourth_term_l3076_307692

theorem proportion_fourth_term 
  (a b c d : ℚ) 
  (h1 : a + b + c = 58)
  (h2 : c = 2/3 * a)
  (h3 : b = 3/4 * a)
  (h4 : a/b = c/d)
  : d = 12 := by
sorry

end proportion_fourth_term_l3076_307692


namespace water_pouring_time_l3076_307646

/-- Proves that pouring 18 gallons at a rate of 1 gallon every 20 seconds takes 6 minutes -/
theorem water_pouring_time (tank_capacity : ℕ) (pour_rate : ℚ) (remaining : ℕ) (poured : ℕ) :
  tank_capacity = 50 →
  pour_rate = 1 / 20 →
  remaining = 32 →
  poured = 18 →
  (poured : ℚ) / pour_rate / 60 = 6 :=
by sorry

end water_pouring_time_l3076_307646


namespace cards_lost_l3076_307696

def initial_cards : ℕ := 88
def remaining_cards : ℕ := 18

theorem cards_lost : initial_cards - remaining_cards = 70 := by
  sorry

end cards_lost_l3076_307696


namespace max_sum_squares_sides_l3076_307683

/-- For any acute-angled triangle with side length a and angle α, 
    the sum of squares of the other two side lengths (b² + c²) 
    is less than or equal to a² / (2 sin²(α/2)). -/
theorem max_sum_squares_sides (a : ℝ) (α : ℝ) (h_acute : 0 < α ∧ α < π / 2) :
  ∀ b c : ℝ, 
  (0 < b ∧ 0 < c) → -- Ensure positive side lengths
  (b^2 + c^2 - 2*b*c*Real.cos α = a^2) → -- Cosine rule
  b^2 + c^2 ≤ a^2 / (2 * Real.sin (α/2)^2) :=
by sorry

end max_sum_squares_sides_l3076_307683


namespace difference_zero_point_eight_and_one_eighth_l3076_307637

theorem difference_zero_point_eight_and_one_eighth : (0.8 : ℝ) - (1 / 8 : ℝ) = 0.675 := by
  sorry

end difference_zero_point_eight_and_one_eighth_l3076_307637


namespace arccos_cos_nine_l3076_307624

/-- The arccosine of the cosine of 9 is equal to 9 modulo 2π. -/
theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 % (2 * Real.pi) := by
  sorry

end arccos_cos_nine_l3076_307624


namespace rectangle_dimension_change_l3076_307680

theorem rectangle_dimension_change 
  (L B : ℝ) -- Original length and breadth
  (L' : ℝ) -- New length
  (h1 : L > 0 ∧ B > 0) -- Positive dimensions
  (h2 : L' * (3 * B) = (3/2) * (L * B)) -- Area increased by 50% and breadth tripled
  : L' = L / 2 := by
sorry

end rectangle_dimension_change_l3076_307680


namespace second_number_value_l3076_307657

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 3 / 4)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end second_number_value_l3076_307657


namespace perpendicular_line_exists_l3076_307632

-- Define a line in 2D space
def Line2D := Set (ℝ × ℝ)

-- Define a point in 2D space
def Point2D := ℝ × ℝ

-- Function to check if a line is perpendicular to another line
def isPerpendicular (l1 l2 : Line2D) : Prop := sorry

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

-- Theorem: For any line and any point, there exists a perpendicular line through that point
theorem perpendicular_line_exists (AB : Line2D) (P : Point2D) : 
  ∃ (l : Line2D), isPerpendicular l AB ∧ isPointOnLine P l := by sorry

end perpendicular_line_exists_l3076_307632


namespace simplify_fraction_with_sqrt_3_l3076_307654

theorem simplify_fraction_with_sqrt_3 :
  (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1/2 := by sorry

end simplify_fraction_with_sqrt_3_l3076_307654


namespace number_of_pencils_l3076_307611

/-- Given that the ratio of pens to pencils is 5 to 6 and there are 8 more pencils than pens,
    prove that the number of pencils is 48. -/
theorem number_of_pencils (pens pencils : ℕ) 
    (h1 : pens * 6 = pencils * 5)  -- ratio of pens to pencils is 5 to 6
    (h2 : pencils = pens + 8)      -- 8 more pencils than pens
    : pencils = 48 := by
  sorry

end number_of_pencils_l3076_307611


namespace expression_bounds_l3076_307679

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1/2 ≤ |2*a - b| / (|a| + |b|) ∧ |2*a - b| / (|a| + |b|) ≤ 1 := by
  sorry

end expression_bounds_l3076_307679


namespace conjecture_counterexample_l3076_307662

theorem conjecture_counterexample : ∃ n : ℕ, 
  (n % 2 = 1 ∧ n > 5) ∧ 
  ¬∃ (p k : ℕ), Prime p ∧ n = p + 2 * k^2 :=
sorry

end conjecture_counterexample_l3076_307662


namespace spencer_burritos_l3076_307677

/-- Represents the number of ways to make burritos with given constraints -/
def burrito_combinations (total_burritos : ℕ) (max_beef : ℕ) (max_chicken : ℕ) (available_wraps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 25 ways to make exactly 5 burritos with the given constraints -/
theorem spencer_burritos : burrito_combinations 5 4 3 5 = 25 := by
  sorry

end spencer_burritos_l3076_307677


namespace total_capacity_is_132000_l3076_307660

/-- The capacity of a train's boxcars -/
def train_capacity (num_red num_blue num_black : ℕ) (black_capacity : ℕ) : ℕ :=
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity

/-- Theorem: The total capacity of the train's boxcars is 132000 pounds -/
theorem total_capacity_is_132000 :
  train_capacity 3 4 7 4000 = 132000 := by
  sorry

end total_capacity_is_132000_l3076_307660


namespace tom_investment_is_3000_l3076_307607

/-- Represents the initial investment problem with Tom and Jose --/
structure InvestmentProblem where
  jose_investment : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the problem parameters --/
def calculate_tom_investment (p : InvestmentProblem) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 --/
theorem tom_investment_is_3000 (p : InvestmentProblem)
  (h1 : p.jose_investment = 45000)
  (h2 : p.jose_months = 10)
  (h3 : p.total_profit = 27000)
  (h4 : p.jose_profit = 15000) :
  calculate_tom_investment p = 3000 :=
sorry

end tom_investment_is_3000_l3076_307607


namespace line_direction_vector_l3076_307602

/-- Given a line passing through two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 6) → p2 = (2, -1) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) →
  b = 5/7 := by
sorry

end line_direction_vector_l3076_307602


namespace third_smallest_four_digit_pascal_l3076_307643

/-- Pascal's triangle as a function from row and column to the value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- The set of all numbers in Pascal's triangle -/
def pascalNumbers : Set ℕ := sorry

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {n ∈ pascalNumbers | 1000 ≤ n ∧ n ≤ 9999}

/-- The third smallest element in a set of natural numbers -/
def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal : 
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end third_smallest_four_digit_pascal_l3076_307643


namespace iggy_monday_run_l3076_307666

/-- Represents the number of miles run on each day of the week --/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total miles run in a week --/
def totalMiles (run : WeeklyRun) : ℝ :=
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday

/-- Represents Iggy's running schedule for the week --/
def iggyRun : WeeklyRun where
  monday := 3  -- This is what we want to prove
  tuesday := 4
  wednesday := 6
  thursday := 8
  friday := 3

/-- Iggy's pace in minutes per mile --/
def pace : ℝ := 10

/-- Total time Iggy spent running in minutes --/
def totalTime : ℝ := 4 * 60

theorem iggy_monday_run :
  iggyRun.monday = 3 ∧ 
  totalMiles iggyRun * pace = totalTime := by
  sorry


end iggy_monday_run_l3076_307666


namespace arithmetic_sum_odd_sequence_l3076_307697

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem arithmetic_sum_odd_sequence :
  let seq := arithmetic_sequence 1 2 11
  (∀ x ∈ seq, is_odd x) ∧
  (seq.length = 11) ∧
  (seq.getLast? = some 21) →
  seq.sum = 121 := by
  sorry

#eval arithmetic_sequence 1 2 11

end arithmetic_sum_odd_sequence_l3076_307697


namespace parallel_lines_imply_a_equals_one_l3076_307693

/-- Two lines are parallel if their slopes are equal and not equal to their y-intercept ratios -/
def are_parallel (a : ℝ) : Prop :=
  a ≠ 0 ∧ a ≠ -1 ∧ (1 / a = a / 1) ∧ (1 / a ≠ (-2*a - 2) / (-a - 1))

/-- Given two lines l₁: x + ay = 2a + 2 and l₂: ax + y = a + 1 are parallel, prove that a = 1 -/
theorem parallel_lines_imply_a_equals_one :
  ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end parallel_lines_imply_a_equals_one_l3076_307693


namespace right_triangle_side_length_l3076_307615

-- Define the right triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180

-- Define the right angle at C
def RightAngleAtC (C : ℝ) : Prop := C = 90

-- Define the sine of angle A
def SineA (sinA : ℝ) : Prop := sinA = Real.sqrt 5 / 3

-- Define the length of side BC
def LengthBC (BC : ℝ) : Prop := BC = 2 * Real.sqrt 5

-- Theorem statement
theorem right_triangle_side_length 
  (A B C AC BC : ℝ) 
  (h_triangle : Triangle A B C) 
  (h_right_angle : RightAngleAtC C) 
  (h_sine_A : SineA (Real.sin (A * π / 180))) 
  (h_BC : LengthBC BC) : 
  AC = 4 := by sorry

end right_triangle_side_length_l3076_307615


namespace opposite_sides_range_l3076_307604

def line_equation (x y a : ℝ) : ℝ := 3 * x - 2 * y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 3 1 a) * (line_equation (-4) 6 a) < 0 ↔ -7 < a ∧ a < 24 := by sorry

end opposite_sides_range_l3076_307604


namespace point_not_in_region_l3076_307685

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end point_not_in_region_l3076_307685


namespace masha_number_is_1001_l3076_307673

/-- Represents the possible operations Vasya could have performed -/
inductive Operation
  | Sum
  | Product

/-- Checks if a number is a valid choice for Sasha or Masha -/
def is_valid_choice (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 2002

/-- Checks if Sasha can determine Masha's number -/
def sasha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a + c = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a * c = 2002

/-- Checks if Masha can determine Sasha's number -/
def masha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c + b = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c * b = 2002

theorem masha_number_is_1001 (a b : ℕ) (op : Operation) :
  is_valid_choice a →
  is_valid_choice b →
  (op = Operation.Sum → a + b = 2002) →
  (op = Operation.Product → a * b = 2002) →
  ¬(sasha_can_determine a b op) →
  ¬(masha_can_determine a b op) →
  b = 1001 := by
  sorry


end masha_number_is_1001_l3076_307673


namespace money_distribution_l3076_307631

/-- Represents the share of money for each person -/
structure Shares :=
  (w : ℚ) (x : ℚ) (y : ℚ) (z : ℚ)

/-- The theorem statement -/
theorem money_distribution (s : Shares) :
  s.w + s.x + s.y + s.z > 0 ∧  -- Ensure total sum is positive
  s.x = 6 * s.w ∧              -- Proportion for x
  s.y = 2 * s.w ∧              -- Proportion for y
  s.z = 4 * s.w ∧              -- Proportion for z
  s.x = s.y + 1500             -- x gets $1500 more than y
  →
  s.w = 375 := by
sorry

end money_distribution_l3076_307631


namespace triangle_ratio_proof_l3076_307649

theorem triangle_ratio_proof (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Angle A is 60°
  A = Real.pi / 3 →
  -- Side b is 1
  b = 1 →
  -- Area of triangle is √3/2
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  -- The sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Triangle inequality
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Prove that the expression equals 2
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end triangle_ratio_proof_l3076_307649


namespace seventh_graders_count_l3076_307656

/-- The number of fifth graders going on the trip -/
def fifth_graders : ℕ := 109

/-- The number of sixth graders going on the trip -/
def sixth_graders : ℕ := 115

/-- The number of teachers chaperoning -/
def teachers : ℕ := 4

/-- The number of parent chaperones per grade -/
def parents_per_grade : ℕ := 2

/-- The number of grades participating -/
def grades : ℕ := 3

/-- The number of buses for the trip -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- Theorem: The number of seventh graders going on the trip is 118 -/
theorem seventh_graders_count : 
  (buses * seats_per_bus) - 
  (fifth_graders + sixth_graders + (teachers + parents_per_grade * grades)) = 118 := by
  sorry

end seventh_graders_count_l3076_307656


namespace store_bottles_l3076_307608

/-- The total number of bottles in a grocery store, given the number of regular and diet soda bottles. -/
def total_bottles (regular_soda : ℕ) (diet_soda : ℕ) : ℕ :=
  regular_soda + diet_soda

/-- Theorem stating that the total number of bottles in the store is 38. -/
theorem store_bottles : total_bottles 30 8 = 38 := by
  sorry

end store_bottles_l3076_307608


namespace smallest_tangent_circle_radius_l3076_307638

theorem smallest_tangent_circle_radius 
  (square_side : ℝ) 
  (semicircle_radius : ℝ) 
  (quarter_circle_radius : ℝ) 
  (h1 : square_side = 4) 
  (h2 : semicircle_radius = 1) 
  (h3 : quarter_circle_radius = 2) : 
  ∃ r : ℝ, r = Real.sqrt 2 - 3/2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → 
    ((x - 2)^2 + y^2 = 1^2 ∨ 
     (x + 2)^2 + y^2 = 1^2 ∨ 
     x^2 + (y - 2)^2 = 1^2 ∨ 
     x^2 + (y + 2)^2 = 1^2) ∧
    x^2 + y^2 = (2 - r)^2) :=
by sorry

end smallest_tangent_circle_radius_l3076_307638


namespace remainder_problem_l3076_307686

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1259 % d = r) (h3 : 1567 % d = r) (h4 : 2257 % d = r) : 
  d - r = 1 := by
  sorry

end remainder_problem_l3076_307686


namespace rectangle_diagonal_l3076_307676

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 23) (h2 : l * w = 120) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end rectangle_diagonal_l3076_307676


namespace unique_solution_power_equation_l3076_307640

theorem unique_solution_power_equation : 
  ∃! (x y z t : ℕ+), 2^y.val + 2^z.val * 5^t.val - 5^x.val = 1 ∧ 
  x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by sorry

end unique_solution_power_equation_l3076_307640


namespace francie_savings_l3076_307668

/-- Calculates Francie's remaining money after saving and spending --/
def franciesRemainingMoney (
  initialWeeklyAllowance : ℕ) 
  (initialWeeks : ℕ)
  (raisedWeeklyAllowance : ℕ)
  (raisedWeeks : ℕ)
  (videoGameCost : ℕ) : ℕ :=
  let totalSavings := initialWeeklyAllowance * initialWeeks + raisedWeeklyAllowance * raisedWeeks
  let remainingAfterClothes := totalSavings / 2
  remainingAfterClothes - videoGameCost

theorem francie_savings : franciesRemainingMoney 5 8 6 6 35 = 3 := by
  sorry

end francie_savings_l3076_307668


namespace smallest_k_and_digit_sum_l3076_307621

-- Define the function to count digits
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the function to sum digits
def sumDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem smallest_k_and_digit_sum :
  ∃ k : ℕ, 
    (k > 0) ∧
    (∀ j : ℕ, j > 0 → j < k → countDigits ((2^j) * (5^300)) < 303) ∧
    (countDigits ((2^k) * (5^300)) = 303) ∧
    (k = 307) ∧
    (sumDigits ((2^k) * (5^300)) = 11) :=
by sorry

end smallest_k_and_digit_sum_l3076_307621


namespace train_speed_l3076_307674

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 16.13204276991174 →
  (train_length + bridge_length) / crossing_time * 3.6 = 54 := by
  sorry

end train_speed_l3076_307674


namespace sandy_fingernail_record_age_l3076_307655

/-- Calculates the age at which Sandy will achieve the world record for longest fingernails -/
theorem sandy_fingernail_record_age 
  (world_record : ℝ)
  (sandy_current_age : ℕ)
  (sandy_current_length : ℝ)
  (growth_rate_per_month : ℝ)
  (h1 : world_record = 26)
  (h2 : sandy_current_age = 12)
  (h3 : sandy_current_length = 2)
  (h4 : growth_rate_per_month = 0.1) :
  sandy_current_age + (world_record - sandy_current_length) / (growth_rate_per_month * 12) = 32 := by
sorry

end sandy_fingernail_record_age_l3076_307655


namespace banana_arrangement_count_l3076_307605

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- The number of 'N's in "BANANA" -/
def num_n : ℕ := 2

/-- The number of 'B's in "BANANA" -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end banana_arrangement_count_l3076_307605


namespace factorization_of_x_squared_minus_four_l3076_307627

theorem factorization_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2) :=
by sorry

end factorization_of_x_squared_minus_four_l3076_307627


namespace circle_area_ratio_l3076_307671

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) : 
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) → 
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end circle_area_ratio_l3076_307671


namespace tan_ratio_problem_l3076_307626

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := by
  sorry

end tan_ratio_problem_l3076_307626


namespace percentage_more_than_6_years_is_21_875_l3076_307663

/-- Represents the employee tenure distribution of a company -/
structure EmployeeTenure where
  less_than_3_years : ℕ
  between_3_and_6_years : ℕ
  more_than_6_years : ℕ

/-- Calculates the percentage of employees who have worked for more than 6 years -/
def percentage_more_than_6_years (e : EmployeeTenure) : ℚ :=
  (e.more_than_6_years : ℚ) / (e.less_than_3_years + e.between_3_and_6_years + e.more_than_6_years) * 100

/-- Proves that the percentage of employees who have worked for more than 6 years is 21.875% -/
theorem percentage_more_than_6_years_is_21_875 (e : EmployeeTenure) 
  (h : ∃ (x : ℕ), e.less_than_3_years = 10 * x ∧ 
                   e.between_3_and_6_years = 15 * x ∧ 
                   e.more_than_6_years = 7 * x) : 
  percentage_more_than_6_years e = 21875 / 1000 := by
  sorry

#eval (21875 : ℚ) / 1000  -- To verify that 21875/1000 = 21.875

end percentage_more_than_6_years_is_21_875_l3076_307663


namespace jesse_blocks_left_l3076_307606

/-- The number of building blocks Jesse has left after constructing various structures --/
def blocks_left (initial : ℕ) (building : ℕ) (farmhouse : ℕ) (fence : ℕ) : ℕ :=
  initial - (building + farmhouse + fence)

/-- Theorem stating that Jesse has 84 blocks left --/
theorem jesse_blocks_left :
  blocks_left 344 80 123 57 = 84 := by
  sorry

end jesse_blocks_left_l3076_307606


namespace printer_task_pages_l3076_307629

theorem printer_task_pages : ∀ (P : ℕ),
  (P / 60 + (P / 60 + 3) = P / 24) →
  (P = 360) :=
by
  sorry

#check printer_task_pages

end printer_task_pages_l3076_307629


namespace circular_track_length_l3076_307622

/-- The length of a circular track given specific overtaking conditions -/
theorem circular_track_length : ∃ (x : ℝ), x > 0 ∧ 279 < x ∧ x < 281 ∧
  ∃ (v_fast v_slow : ℝ), v_fast > v_slow ∧ v_fast > 0 ∧ v_slow > 0 ∧
  (150 / (x - 150) = (x + 100) / (x + 50)) := by
  sorry

end circular_track_length_l3076_307622


namespace water_heater_theorem_l3076_307636

/-- Calculates the total amount of water in two water heaters -/
def total_water (wallace_capacity : ℚ) : ℚ :=
  let catherine_capacity := wallace_capacity / 2
  let wallace_water := (3 / 4) * wallace_capacity
  let catherine_water := (3 / 4) * catherine_capacity
  wallace_water + catherine_water

/-- Theorem stating that given the conditions, the total water is 45 gallons -/
theorem water_heater_theorem (wallace_capacity : ℚ) 
  (h1 : wallace_capacity = 40) : total_water wallace_capacity = 45 := by
  sorry

#eval total_water 40

end water_heater_theorem_l3076_307636


namespace initial_girls_count_l3076_307688

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 : ℚ) / total = 2/5 →  -- After changes, 40% are girls
  initial_girls = 15 := by
sorry

end initial_girls_count_l3076_307688


namespace wall_thickness_is_5cm_l3076_307610

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in meters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick in cubic centimeters -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the total volume of bricks in cubic centimeters -/
def totalBrickVolume (b : BrickDimensions) (n : ℝ) : ℝ :=
  brickVolume b * n

/-- Calculates the area of the wall's face in square centimeters -/
def wallFaceArea (w : WallDimensions) : ℝ :=
  w.length * w.height * 10000 -- Convert m² to cm²

/-- The main theorem stating the wall thickness -/
theorem wall_thickness_is_5cm 
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℝ) :
  brick.length = 25 ∧ 
  brick.width = 11 ∧ 
  brick.height = 6 ∧
  wall.length = 8 ∧
  wall.height = 1 ∧
  num_bricks = 242.42424242424244 →
  wall.thickness = 5 := by
sorry

end wall_thickness_is_5cm_l3076_307610


namespace greatest_c_value_l3076_307650

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end greatest_c_value_l3076_307650


namespace spheres_radius_is_correct_l3076_307670

/-- Right circular cone with given dimensions -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Sphere inside the cone -/
structure Sphere where
  radius : ℝ

/-- Configuration of three spheres inside a cone -/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  centerPlaneHeight : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : SpheresInCone where
  cone := { baseRadius := 4, height := 15 }
  sphere := { radius := 1.5 }  -- We use the correct answer here
  centerPlaneHeight := 2.5     -- This is r + 1, where r = 1.5

/-- Predicate to check if the spheres are tangent to each other, the base, and the side of the cone -/
def areTangent (config : SpheresInCone) : Prop :=
  -- The actual tangency conditions would be complex to express precisely,
  -- so we use a placeholder predicate
  True

/-- Theorem stating that the given configuration satisfies the problem conditions -/
theorem spheres_radius_is_correct (config : SpheresInCone) :
  config.cone.baseRadius = 4 ∧
  config.cone.height = 15 ∧
  config.centerPlaneHeight = config.sphere.radius + 1 ∧
  areTangent config →
  config.sphere.radius = 1.5 :=
by
  sorry

#check spheres_radius_is_correct

end spheres_radius_is_correct_l3076_307670


namespace race_time_proof_l3076_307620

/-- Represents the time taken by runner A to complete the race -/
def time_A : ℝ := 235

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the distance by which A beats B in meters -/
def distance_difference : ℝ := 60

/-- Represents the time difference by which A beats B in seconds -/
def time_difference : ℝ := 15

/-- Theorem stating that given the race conditions, runner A completes the race in 235 seconds -/
theorem race_time_proof :
  (race_length / time_A = (race_length - distance_difference) / time_A) ∧
  (race_length / time_A = race_length / (time_A + time_difference)) →
  time_A = 235 := by
  sorry

end race_time_proof_l3076_307620


namespace triangle_problem_l3076_307616

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The cosine of an angle in a triangle -/
def cosine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The sine of an angle in a triangle -/
def sine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - (3/4) * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (k : ℝ), sine t 1 = k - (sine t 2) ∧ sine t 3 = k + (sine t 2)) :
  cosine t 2 = 5/8 ∧ area t = (3 * Real.sqrt 39) / 4 := by
  sorry

end triangle_problem_l3076_307616


namespace jane_brown_sheets_l3076_307603

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets (total : ℕ) (yellow : ℕ) : ℕ := total - yellow

/-- Proof that Jane has 28 old, brown sheets of drawing paper -/
theorem jane_brown_sheets : brown_sheets 55 27 = 28 := by
  sorry

end jane_brown_sheets_l3076_307603


namespace common_chord_circles_l3076_307617

/-- Given two circles with equations x^2 + (y - 3/2)^2 = 25/4 and x^2 + y^2 = m,
    if they have a common chord passing through the point (0, 3/2), then m = 17/2. -/
theorem common_chord_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 3/2)^2 = 25/4 ∧ x^2 + y^2 = m) ∧ 
  (∃ (x : ℝ), x^2 + (3/2 - 3/2)^2 = 25/4 ∧ x^2 + (3/2)^2 = m) →
  m = 17/2 := by
  sorry

end common_chord_circles_l3076_307617


namespace both_heads_prob_l3076_307667

/-- Represents the outcome of flipping two coins simultaneously -/
inductive CoinFlip
| HH -- Both heads
| HT -- First head, second tail
| TH -- First tail, second head
| TT -- Both tails

/-- The probability of getting a specific outcome when flipping two fair coins -/
def flip_prob : CoinFlip → ℚ
| CoinFlip.HH => 1/4
| CoinFlip.HT => 1/4
| CoinFlip.TH => 1/4
| CoinFlip.TT => 1/4

/-- The process of flipping coins until at least one head appears -/
def flip_until_head : ℕ → ℚ
| 0 => flip_prob CoinFlip.HH
| (n+1) => flip_prob CoinFlip.TT * flip_until_head n

/-- The theorem stating the probability of both coins showing heads when the process stops -/
theorem both_heads_prob : (∑' n, flip_until_head n) = 1/3 :=
sorry


end both_heads_prob_l3076_307667


namespace problem_solution_l3076_307633

theorem problem_solution (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by
  sorry

end problem_solution_l3076_307633
