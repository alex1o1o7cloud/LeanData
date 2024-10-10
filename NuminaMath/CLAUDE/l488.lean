import Mathlib

namespace quadratic_no_real_roots_l488_48872

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), 2 * x^2 - 5 * x + 6 ≠ 0 :=
sorry

end quadratic_no_real_roots_l488_48872


namespace taxi_trip_length_l488_48884

theorem taxi_trip_length 
  (initial_fee : ℝ) 
  (additional_charge : ℝ) 
  (segment_length : ℝ) 
  (total_charge : ℝ) : 
  initial_fee = 2.25 →
  additional_charge = 0.15 →
  segment_length = 2/5 →
  total_charge = 3.60 →
  ∃ (trip_length : ℝ), 
    trip_length = 3.6 ∧ 
    total_charge = initial_fee + (trip_length / segment_length) * additional_charge :=
by sorry

end taxi_trip_length_l488_48884


namespace train_speed_calculation_l488_48811

/-- Given a train that crosses a platform and passes a stationary man, calculate its speed -/
theorem train_speed_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 220 →
  platform_crossing_time = 30 →
  man_passing_time = 19 →
  ∃ (train_speed : ℝ), train_speed = 72 := by
  sorry

end train_speed_calculation_l488_48811


namespace vaccine_cost_l488_48898

theorem vaccine_cost (num_vaccines : ℕ) (doctor_visit_cost : ℝ) 
  (insurance_coverage : ℝ) (trip_cost : ℝ) (total_payment : ℝ) :
  num_vaccines = 10 ∧ 
  doctor_visit_cost = 250 ∧ 
  insurance_coverage = 0.8 ∧ 
  trip_cost = 1200 ∧ 
  total_payment = 1340 →
  (total_payment - trip_cost - (1 - insurance_coverage) * doctor_visit_cost) / 
  ((1 - insurance_coverage) * num_vaccines) = 45 := by
  sorry

end vaccine_cost_l488_48898


namespace bobbit_worm_days_l488_48840

/-- The number of days the Bobbit worm was in the aquarium before James added more fish -/
def days_before_adding : ℕ := sorry

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish the Bobbit worm eats per day -/
def fish_eaten_per_day : ℕ := 2

/-- The number of fish James adds to the aquarium -/
def fish_added : ℕ := 8

/-- The number of days between adding fish and discovering the Bobbit worm -/
def days_after_adding : ℕ := 7

/-- The final number of fish in the aquarium when James discovers the Bobbit worm -/
def final_fish : ℕ := 26

theorem bobbit_worm_days : 
  initial_fish - (fish_eaten_per_day * days_before_adding) + fish_added - (fish_eaten_per_day * days_after_adding) = final_fish ∧
  days_before_adding = 14 := by sorry

end bobbit_worm_days_l488_48840


namespace correct_calculation_l488_48843

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end correct_calculation_l488_48843


namespace west_distance_negative_l488_48869

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function to record distance based on direction
def recordDistance (distance : ℝ) (direction : Direction) : ℝ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_distance_negative (d : ℝ) :
  d > 0 → recordDistance d Direction.East = d → recordDistance d Direction.West = -d :=
by
  sorry

end west_distance_negative_l488_48869


namespace ellipse_circle_tangent_property_l488_48864

/-- Given an ellipse and a circle, prove a property of tangents from a point on the ellipse to the circle -/
theorem ellipse_circle_tangent_property
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (P : ℝ × ℝ) (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (hP_not_vertex : P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧ P ≠ (0, b) ∧ P ≠ (0, -b))
  (A B : ℝ × ℝ)
  (hA : A.1^2 + A.2^2 = b^2)
  (hB : B.1^2 + B.2^2 = b^2)
  (hPA : P.1 * A.1 + P.2 * A.2 = b^2)
  (hPB : P.1 * B.1 + P.2 * B.2 = b^2)
  (M : ℝ × ℝ) (hM : M.2 = 0 ∧ M.1 * P.1 = b^2)
  (N : ℝ × ℝ) (hN : N.1 = 0 ∧ N.2 * P.2 = b^2) :
  a^2 / (N.2^2) + b^2 / (M.1^2) = a^2 / b^2 :=
sorry

end ellipse_circle_tangent_property_l488_48864


namespace congruence_problem_l488_48849

theorem congruence_problem (x : ℤ) :
  x ≡ 3 [ZMOD 7] →
  x^2 ≡ 44 [ZMOD (7^2)] →
  x^3 ≡ 111 [ZMOD (7^3)] →
  x ≡ 17 [ZMOD 343] := by
sorry

end congruence_problem_l488_48849


namespace expression_simplification_l488_48895

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 := by
  sorry

end expression_simplification_l488_48895


namespace sum_of_roots_l488_48870

theorem sum_of_roots (k c : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : 6 * x₁^2 - k * x₁ = c)
  (h₂ : 6 * x₂^2 - k * x₂ = c) :
  x₁ + x₂ = k / 6 := by
sorry

end sum_of_roots_l488_48870


namespace paper_clip_distribution_l488_48845

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes : ℕ) : 
  total_clips = 81 → 
  clips_per_box = 9 → 
  total_clips = boxes * clips_per_box → 
  boxes = 9 := by
sorry

end paper_clip_distribution_l488_48845


namespace adoption_time_proof_l488_48837

/-- The number of days required to adopt all puppies in a pet shelter -/
def adoptionDays (initialPuppies additionalPuppies adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem: It takes 7 days to adopt all puppies given the specified conditions -/
theorem adoption_time_proof :
  adoptionDays 9 12 3 = 7 := by
  sorry

end adoption_time_proof_l488_48837


namespace reciprocal_of_negative_eight_l488_48887

theorem reciprocal_of_negative_eight :
  ∃ x : ℚ, x * (-8) = 1 ∧ x = -1/8 := by
  sorry

end reciprocal_of_negative_eight_l488_48887


namespace like_terms_exponents_l488_48833

theorem like_terms_exponents (a b x y : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 5 * a^(|x|) * b^2 = k * (-0.2 * a^3 * b^(|y|))) → 
  |x| = 3 ∧ |y| = 2 :=
by sorry

end like_terms_exponents_l488_48833


namespace monthly_income_calculation_l488_48839

/-- Proves that if 22% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 17272.73. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 22)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 17272.73 := by
  sorry

end monthly_income_calculation_l488_48839


namespace proposition_equivalences_l488_48820

-- Define opposite numbers
def opposite (x y : ℝ) : Prop := x = -y

-- Define having real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem proposition_equivalences :
  -- Converse of "If x+y=0, then x and y are opposite numbers"
  (∀ x y : ℝ, opposite x y → x + y = 0) ∧
  -- Contrapositive of "If q ≤ 1, then x^2+2x+q=0 has real roots"
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  -- Existence of α and β satisfying the trigonometric equation
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end proposition_equivalences_l488_48820


namespace pentagon_arrangement_exists_l488_48881

/-- Represents a pentagon arrangement of natural numbers -/
def PentagonArrangement := Fin 5 → ℕ

/-- Checks if two numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if two numbers have a common divisor greater than 1 -/
def have_common_divisor (a b : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

/-- Checks if the given arrangement satisfies the conditions -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  (∀ i : Fin 5, are_coprime (arr i) (arr ((i + 1) % 5))) ∧
  (∀ i : Fin 5, have_common_divisor (arr i) (arr ((i + 2) % 5)))

/-- The main theorem: there exists a valid pentagon arrangement -/
theorem pentagon_arrangement_exists : ∃ arr : PentagonArrangement, is_valid_arrangement arr :=
sorry

end pentagon_arrangement_exists_l488_48881


namespace complement_intersection_theorem_l488_48836

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 5}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end complement_intersection_theorem_l488_48836


namespace intersection_of_lines_AB_CD_l488_48817

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := A
  let b := B
  let c := C
  let d := D
  (20, -18, 11)

/-- Theorem stating that the intersection point of lines AB and CD is (20, -18, 11) --/
theorem intersection_of_lines_AB_CD :
  let A : ℝ × ℝ × ℝ := (8, -6, 5)
  let B : ℝ × ℝ × ℝ := (18, -16, 10)
  let C : ℝ × ℝ × ℝ := (-4, 6, -12)
  let D : ℝ × ℝ × ℝ := (4, -4, 8)
  intersection_point A B C D = (20, -18, 11) := by
  sorry

#check intersection_of_lines_AB_CD

end intersection_of_lines_AB_CD_l488_48817


namespace eight_entrepreneurs_not_attending_l488_48855

/-- The number of entrepreneurs who did not attend either session -/
def entrepreneurs_not_attending (total : ℕ) (digital : ℕ) (ecommerce : ℕ) (both : ℕ) : ℕ :=
  total - (digital + ecommerce - both)

/-- Theorem: Given the specified numbers of entrepreneurs, prove that 8 did not attend either session -/
theorem eight_entrepreneurs_not_attending :
  entrepreneurs_not_attending 40 22 18 8 = 8 := by
  sorry

end eight_entrepreneurs_not_attending_l488_48855


namespace perspective_properties_l488_48871

-- Define a type for perspective drawings
def PerspectiveDrawing : Type := sorry

-- Define a function to represent the perspective drawing of a square
def perspectiveSquare : PerspectiveDrawing → Bool := sorry

-- Define a function to represent the perspective drawing of intersecting lines
def perspectiveIntersectingLines : PerspectiveDrawing → Bool := sorry

-- Define a function to represent the perspective drawing of perpendicular lines
def perspectivePerpendicularLines : PerspectiveDrawing → Bool := sorry

theorem perspective_properties :
  ∃ (d : PerspectiveDrawing),
    (¬ perspectiveSquare d) ∧
    (¬ perspectiveIntersectingLines d) ∧
    (¬ perspectivePerpendicularLines d) := by
  sorry

end perspective_properties_l488_48871


namespace largest_common_value_l488_48866

def first_progression (n : ℕ) : ℕ := 4 + 5 * n

def second_progression (m : ℕ) : ℕ := 5 + 10 * m

theorem largest_common_value :
  ∃ (n m : ℕ),
    first_progression n = second_progression m ∧
    first_progression n < 1000 ∧
    first_progression n ≡ 1 [MOD 4] ∧
    ∀ (k l : ℕ),
      first_progression k = second_progression l →
      first_progression k < 1000 →
      first_progression k ≡ 1 [MOD 4] →
      first_progression k ≤ first_progression n :=
by
  sorry

end largest_common_value_l488_48866


namespace minimum_a_value_l488_48893

def set_A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 4/5}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

theorem minimum_a_value (a : ℝ) (h : set_A ⊆ set_B a) : a ≥ 2 := by
  sorry

end minimum_a_value_l488_48893


namespace line_equation_solution_l488_48873

/-- The line x = k intersects y = x^2 + 4x + 4 and y = mx + b at two points 4 units apart -/
def intersectionCondition (m b k : ℝ) : Prop :=
  ∃ k, |k^2 + 4*k + 4 - (m*k + b)| = 4

/-- The line y = mx + b passes through the point (2, 8) -/
def passesThroughPoint (m b : ℝ) : Prop :=
  8 = 2*m + b

/-- b is not equal to 0 -/
def bNonZero (b : ℝ) : Prop :=
  b ≠ 0

/-- The theorem stating that given the conditions, the unique solution for the line equation is y = 8x - 8 -/
theorem line_equation_solution (m b : ℝ) :
  (∃ k, intersectionCondition m b k) →
  passesThroughPoint m b →
  bNonZero b →
  m = 8 ∧ b = -8 :=
by sorry

end line_equation_solution_l488_48873


namespace license_plate_increase_l488_48859

def old_plates : ℕ := 26^3 * 10^3
def new_plates_a : ℕ := 26^2 * 10^4
def new_plates_b : ℕ := 26^4 * 10^2
def avg_new_plates : ℚ := (new_plates_a + new_plates_b) / 2

theorem license_plate_increase : 
  (avg_new_plates : ℚ) / old_plates = 468 / 10 := by sorry

end license_plate_increase_l488_48859


namespace correct_matching_probability_l488_48822

/-- The number of celebrities and baby pictures --/
def n : ℕ := 3

/-- The total number of possible arrangements --/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements --/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures --/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_matching_probability :
  probability = 1 / 6 := by sorry

end correct_matching_probability_l488_48822


namespace a_3_equals_zero_l488_48832

theorem a_3_equals_zero (a : ℕ → ℝ) (h : ∀ n, a n = Real.sin (n * π / 3)) : a 3 = 0 := by
  sorry

end a_3_equals_zero_l488_48832


namespace total_coins_l488_48888

theorem total_coins (quarters_piles dimes_piles nickels_piles pennies_piles : ℕ)
  (quarters_per_pile dimes_per_pile nickels_per_pile pennies_per_pile : ℕ)
  (h1 : quarters_piles = 5)
  (h2 : dimes_piles = 5)
  (h3 : nickels_piles = 3)
  (h4 : pennies_piles = 4)
  (h5 : quarters_per_pile = 3)
  (h6 : dimes_per_pile = 3)
  (h7 : nickels_per_pile = 4)
  (h8 : pennies_per_pile = 5) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 62 :=
by sorry

end total_coins_l488_48888


namespace janes_bagels_l488_48824

theorem janes_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 := by
sorry

end janes_bagels_l488_48824


namespace suzy_twice_mary_age_l488_48862

/-- The number of years in the future when Suzy will be twice Mary's age -/
def future_years : ℕ := 4

/-- Suzy's current age -/
def suzy_age : ℕ := 20

/-- Mary's current age -/
def mary_age : ℕ := 8

/-- Theorem stating that in 'future_years', Suzy will be twice Mary's age -/
theorem suzy_twice_mary_age : 
  suzy_age + future_years = 2 * (mary_age + future_years) := by sorry

end suzy_twice_mary_age_l488_48862


namespace sum_of_coefficients_l488_48844

-- Define the polynomial
def polynomial (a b c d : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the root
def root : ℂ := 2 + Complex.I

-- Theorem statement
theorem sum_of_coefficients (a b c d : ℝ) : 
  polynomial a b c d root = 0 → a + b + c + d = 10 :=
by sorry

end sum_of_coefficients_l488_48844


namespace phone_profit_optimization_l488_48861

/-- Represents the profit calculation and optimization problem for two types of phones. -/
theorem phone_profit_optimization
  (profit_A_B : ℕ → ℕ → ℝ)
  (total_phones : ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (h3 : total_phones = 20)
  (h4 : ∀ x y, x + y = total_phones → y ≤ 2 / 3 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    y ≤ 2 / 3 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → a ≥ 0 → b ≥ 0 →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 :=
by sorry

end phone_profit_optimization_l488_48861


namespace intersection_point_sum_l488_48800

theorem intersection_point_sum (a' b' : ℚ) : 
  (2 = (1/3) * 4 + a') ∧ (4 = (1/3) * 2 + b') → a' + b' = 4 := by
  sorry

end intersection_point_sum_l488_48800


namespace initial_fish_count_l488_48857

theorem initial_fish_count (x : ℕ) : 
  x - 50 - (x - 50) / 3 + 200 = 300 → x = 200 := by
  sorry

end initial_fish_count_l488_48857


namespace complex_fraction_calculation_l488_48854

theorem complex_fraction_calculation : 
  (6 + 3/5 - (17/2 - 1/3) / (7/2)) * (2 + 5/18 + 11/12) = 368/27 := by
  sorry

end complex_fraction_calculation_l488_48854


namespace feathers_per_crown_calculation_l488_48835

/-- Given a total number of feathers and a number of crowns, 
    calculate the number of feathers per crown. -/
def feathers_per_crown (total_feathers : ℕ) (num_crowns : ℕ) : ℕ :=
  (total_feathers + num_crowns - 1) / num_crowns

/-- Theorem stating that given 6538 feathers and 934 crowns, 
    the number of feathers per crown is 7. -/
theorem feathers_per_crown_calculation :
  feathers_per_crown 6538 934 = 7 := by
  sorry


end feathers_per_crown_calculation_l488_48835


namespace bike_rental_cost_theorem_l488_48875

/-- The fee structure and rental details for a bicycle rental service. -/
structure BikeRental where
  fee_per_30min : ℕ  -- Fee in won for 30 minutes
  num_bikes : ℕ      -- Number of bikes rented
  duration_hours : ℕ -- Duration of rental in hours
  num_people : ℕ     -- Number of people splitting the cost

/-- Calculate the cost per person for a bike rental. -/
def cost_per_person (rental : BikeRental) : ℕ :=
  let total_cost := rental.fee_per_30min * 2 * rental.duration_hours * rental.num_bikes
  total_cost / rental.num_people

/-- Theorem stating that under the given conditions, each person pays 16000 won. -/
theorem bike_rental_cost_theorem (rental : BikeRental) 
  (h1 : rental.fee_per_30min = 4000)
  (h2 : rental.num_bikes = 4)
  (h3 : rental.duration_hours = 3)
  (h4 : rental.num_people = 6) : 
  cost_per_person rental = 16000 := by
  sorry

end bike_rental_cost_theorem_l488_48875


namespace matching_shoes_probability_l488_48806

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select 2 shoes out of the total -/
def total_selections : ℕ := (total_shoes.choose 2)

/-- The number of ways to select a matching pair -/
def matching_selections : ℕ := num_pairs

/-- The probability of selecting a matching pair -/
def probability_matching : ℚ := matching_selections / total_selections

theorem matching_shoes_probability : probability_matching = 1 / 9 := by
  sorry

end matching_shoes_probability_l488_48806


namespace negation_of_proposition_l488_48831

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by sorry

end negation_of_proposition_l488_48831


namespace tangent_ball_prism_area_relation_l488_48856

/-- A quadrangular prism with a small ball tangent to each face -/
structure TangentBallPrism where
  S₁ : ℝ  -- Area of the upper base
  S₂ : ℝ  -- Area of the lower base
  S : ℝ   -- Lateral surface area
  h₁ : 0 < S₁  -- S₁ is positive
  h₂ : 0 < S₂  -- S₂ is positive
  h₃ : 0 < S   -- S is positive

/-- The relationship between the lateral surface area and the base areas -/
theorem tangent_ball_prism_area_relation (p : TangentBallPrism) : 
  Real.sqrt p.S = Real.sqrt p.S₁ + Real.sqrt p.S₂ := by
  sorry

end tangent_ball_prism_area_relation_l488_48856


namespace no_integral_solution_l488_48828

theorem no_integral_solution : ¬∃ (n m : ℤ), n^2 + (n+1)^2 + (n+2)^2 = m^2 := by
  sorry

end no_integral_solution_l488_48828


namespace smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l488_48867

theorem smallest_addition_for_divisibility (n : ℕ) : 
  (∃ (k : ℕ), k < 9 ∧ (n + k) % 9 = 0) → 
  (∃ (m : ℕ), m < 9 ∧ (n + m) % 9 = 0 ∧ ∀ (l : ℕ), l < m → (n + l) % 9 ≠ 0) :=
by sorry

theorem solution_for_27461 : 
  ∃ (k : ℕ), k < 9 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

theorem answer_is_seven : 
  ∃ (k : ℕ), k = 7 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

end smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l488_48867


namespace intersection_A_not_B_range_of_a_l488_48874

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B
def not_B : Set ℝ := {x | x < 2}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem 1: A ∩ (¬ᵣB) = {x | 1 < x < 2}
theorem intersection_A_not_B : A ∩ not_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: The range of a is [1, +∞) when A ∩ C = C
theorem range_of_a (a : ℝ) : A ∩ C a = C a ↔ a ≥ 1 := by sorry

end intersection_A_not_B_range_of_a_l488_48874


namespace grid_completion_count_l488_48877

/-- Represents a 2x3 grid with one fixed R and 5 remaining squares --/
def Grid := Fin 2 → Fin 3 → Fin 3

/-- Checks if two adjacent cells in the grid have the same value --/
def has_adjacent_match (g : Grid) : Prop :=
  ∃ i j, (g i j = g i (j + 1)) ∨ 
         (g i j = g (i + 1) j)

/-- The number of ways to fill the grid --/
def total_configurations : ℕ := 3^5

/-- The number of valid configurations without adjacent matches --/
def valid_configurations : ℕ := 18

theorem grid_completion_count :
  (total_configurations - valid_configurations : ℕ) = 225 :=
sorry

end grid_completion_count_l488_48877


namespace base_number_proof_l488_48829

theorem base_number_proof (e : ℕ) (x : ℕ) : 
  e = x^19 ∧ e % 10 = 7 → x = 3 :=
by sorry

end base_number_proof_l488_48829


namespace kylie_coins_left_l488_48882

/-- Calculates the number of coins Kylie has left after giving half to Laura -/
def kyliesRemainingCoins (piggyBank : ℕ) (brotherCoins : ℕ) (sofaCoins : ℕ) : ℕ :=
  let fatherCoins := 2 * brotherCoins
  let totalCoins := piggyBank + brotherCoins + fatherCoins + sofaCoins
  totalCoins - (totalCoins / 2)

/-- Theorem stating that Kylie has 62 coins left -/
theorem kylie_coins_left :
  kyliesRemainingCoins 30 26 15 = 62 := by
  sorry

#eval kyliesRemainingCoins 30 26 15

end kylie_coins_left_l488_48882


namespace circle_equation_l488_48827

/-- The standard equation of a circle with center on y = 2x and tangent to x-axis at (-1, 0) -/
theorem circle_equation : ∃ (h k : ℝ), 
  (h = -1 ∧ k = -2) ∧  -- Center on y = 2x
  ((x : ℝ) + 1)^2 + ((y : ℝ) + 2)^2 = 4 ∧  -- Standard equation
  (∀ (x y : ℝ), y = 2*x → (x - h)^2 + (y - k)^2 = 4) ∧  -- Center on y = 2x
  ((-1 : ℝ) - h)^2 + (0 - k)^2 = 4  -- Tangent to x-axis at (-1, 0)
  := by sorry

end circle_equation_l488_48827


namespace parabolic_arch_bridge_width_l488_48810

/-- Parabolic arch bridge problem -/
theorem parabolic_arch_bridge_width 
  (a : ℝ) 
  (h1 : a = -8) 
  (h2 : 4^2 = a * (-2)) 
  : let new_y := -3/2
    let new_x := Real.sqrt (a * new_y)
    2 * new_x = 4 * Real.sqrt 3 := by
  sorry

end parabolic_arch_bridge_width_l488_48810


namespace third_flip_probability_is_one_sixth_l488_48853

/-- Represents the "Treasure Box" game in the "Lucky 52" program --/
structure TreasureBoxGame where
  total_logos : ℕ
  winning_logos : ℕ
  flips : ℕ
  flipped_winning_logos : ℕ

/-- The probability of winning on the third flip in the Treasure Box game --/
def third_flip_probability (game : TreasureBoxGame) : ℚ :=
  let remaining_logos := game.total_logos - game.flipped_winning_logos
  let remaining_winning_logos := game.winning_logos - game.flipped_winning_logos
  remaining_winning_logos / remaining_logos

/-- Theorem stating the probability of winning on the third flip --/
theorem third_flip_probability_is_one_sixth 
  (game : TreasureBoxGame) 
  (h1 : game.total_logos = 20)
  (h2 : game.winning_logos = 5)
  (h3 : game.flips = 3)
  (h4 : game.flipped_winning_logos = 2) :
  third_flip_probability game = 1/6 := by
  sorry


end third_flip_probability_is_one_sixth_l488_48853


namespace lucy_money_problem_l488_48801

theorem lucy_money_problem (initial_amount : ℝ) : 
  let doubled := 2 * initial_amount
  let after_loss := doubled * (2/3)
  let after_spending := after_loss * (3/4)
  after_spending = 15 → initial_amount = 15 := by
sorry

end lucy_money_problem_l488_48801


namespace f_inequality_and_range_l488_48890

noncomputable def f (x : ℝ) := 1 - Real.exp (-x)

theorem f_inequality_and_range :
  (∀ x > -1, f x ≥ x / (x + 1)) ∧
  (Set.Icc (0 : ℝ) (1/2) = {a | ∀ x ≥ 0, f x ≤ x / (a * x + 1)}) :=
sorry

end f_inequality_and_range_l488_48890


namespace physical_fitness_test_probability_l488_48892

theorem physical_fitness_test_probability 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (selected_students : ℕ) :
  total_students = male_students + female_students →
  male_students = 3 →
  female_students = 2 →
  selected_students = 2 →
  (Nat.choose male_students 1 * Nat.choose female_students 1) / 
  Nat.choose total_students selected_students = 3 / 5 := by
sorry

end physical_fitness_test_probability_l488_48892


namespace unique_k_satisfying_conditions_l488_48886

/-- A sequence of binomial coefficients forms an arithmetic progression -/
def is_arithmetic_progression (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, i < k → (n.choose (j + i + 1) : ℤ) - (n.choose (j + i) : ℤ) = d

/-- Condition for part a) -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, ¬∃ j : ℕ, j ≤ n - k + 1 ∧ is_arithmetic_progression n j k

/-- Condition for part b) -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_progression n j (k - 1)

/-- The main theorem -/
theorem unique_k_satisfying_conditions :
  ∃! k : ℕ, k > 0 ∧ condition_a k ∧ condition_b k :=
sorry

end unique_k_satisfying_conditions_l488_48886


namespace prob_three_same_suit_standard_deck_l488_48847

/-- A standard deck of cards --/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Standard 52-card deck --/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4,
    black_suits := 2,
    red_suits := 2 }

/-- Probability of drawing three cards of the same specific suit --/
def prob_three_same_suit (d : Deck) : Rat :=
  (d.ranks * (d.ranks - 1) * (d.ranks - 2)) / (d.cards * (d.cards - 1) * (d.cards - 2))

/-- Theorem stating the probability of drawing three cards of the same specific suit from a standard deck --/
theorem prob_three_same_suit_standard_deck :
  prob_three_same_suit standard_deck = 11 / 850 := by
  sorry

end prob_three_same_suit_standard_deck_l488_48847


namespace total_cards_l488_48897

theorem total_cards (hockey_cards : ℕ) 
  (h1 : hockey_cards = 200)
  (h2 : ∃ football_cards : ℕ, football_cards = 4 * hockey_cards)
  (h3 : ∃ baseball_cards : ℕ, baseball_cards = football_cards - 50) :
  ∃ total_cards : ℕ, total_cards = hockey_cards + football_cards + baseball_cards ∧ total_cards = 1750 :=
by
  sorry

end total_cards_l488_48897


namespace tristan_saturday_study_time_l488_48878

/-- Calculates Tristan's study hours on Saturday given his study schedule --/
def tristanSaturdayStudyHours (mondayHours : ℕ) (weekdayHours : ℕ) (totalWeekHours : ℕ) : ℕ :=
  let tuesdayHours := 2 * mondayHours
  let wednesdayToFridayHours := 3 * weekdayHours
  let mondayToFridayHours := mondayHours + tuesdayHours + wednesdayToFridayHours
  let remainingHours := totalWeekHours - mondayToFridayHours
  remainingHours / 2

/-- Theorem: Given Tristan's study schedule, he studies for 2 hours on Saturday --/
theorem tristan_saturday_study_time :
  tristanSaturdayStudyHours 4 3 25 = 2 := by
  sorry

end tristan_saturday_study_time_l488_48878


namespace arithmetic_sequence_sum_l488_48826

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 2 and a₂ + a₃ = 13,
    prove that a₄ + a₅ + a₆ = 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : IsArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_a2_a3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end arithmetic_sequence_sum_l488_48826


namespace floor_plus_self_equals_ten_point_three_l488_48812

theorem floor_plus_self_equals_ten_point_three (r : ℝ) :
  (⌊r⌋ : ℝ) + r = 10.3 → r = 5.3 := by
  sorry

end floor_plus_self_equals_ten_point_three_l488_48812


namespace trigonometric_equation_solution_l488_48858

theorem trigonometric_equation_solution (t : ℝ) : 
  (Real.sin (2 * t))^6 + (Real.cos (2 * t))^6 = 
    3/2 * ((Real.sin (2 * t))^4 + (Real.cos (2 * t))^4) + 1/2 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = π * (2 * k + 1)) ∨ 
  (∃ n : ℤ, t = π/2 * (4 * n - 1)) := by
sorry

end trigonometric_equation_solution_l488_48858


namespace total_votes_polled_l488_48825

/-- Represents the total number of votes polled in an election --/
def total_votes : ℕ := sorry

/-- Represents the number of valid votes received by candidate B --/
def votes_B : ℕ := 2509

/-- Theorem stating the total number of votes polled in the election --/
theorem total_votes_polled :
  (total_votes : ℚ) * (80 : ℚ) / 100 = 
    (votes_B : ℚ) + (votes_B : ℚ) + (total_votes : ℚ) * (15 : ℚ) / 100 ∧
  total_votes = 7720 :=
sorry

end total_votes_polled_l488_48825


namespace gcd_problem_l488_48891

/-- The operation * represents the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: The GCD of ((12 * 16) * (18 * 12)) is 2 -/
theorem gcd_problem : gcd_op (gcd_op (gcd_op 12 16) (gcd_op 18 12)) 2 = 2 := by
  sorry

end gcd_problem_l488_48891


namespace max_value_constrained_expression_l488_48803

theorem max_value_constrained_expression :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 →
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
by sorry

end max_value_constrained_expression_l488_48803


namespace special_function_form_l488_48814

/-- A bijective, monotonic function from ℝ to ℝ satisfying f(t) + f⁻¹(t) = 2t for all t ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ 
  Monotone f ∧ 
  ∀ t : ℝ, f t + Function.invFun f t = 2 * t

/-- The theorem stating that any SpecialFunction is of the form f(x) = x + c for some constant c -/
theorem special_function_form (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by sorry

end special_function_form_l488_48814


namespace correct_calculation_l488_48804

theorem correct_calculation (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end correct_calculation_l488_48804


namespace square_fraction_integers_l488_48896

theorem square_fraction_integers (n : ℕ) : n > 1 ∧ ∃ (k : ℕ), k > 0 ∧ (n^2 + 7*n + 136) / (n - 1) = k^2 ↔ n = 5 ∨ n = 37 := by
  sorry

end square_fraction_integers_l488_48896


namespace division_problem_l488_48808

/-- Given a division problem with quotient, divisor, and remainder, calculate the dividend -/
theorem division_problem (quotient divisor remainder : ℕ) (h1 : quotient = 256) (h2 : divisor = 3892) (h3 : remainder = 354) :
  divisor * quotient + remainder = 996706 := by
  sorry

end division_problem_l488_48808


namespace sandy_change_is_three_l488_48852

/-- Represents the cost and quantity of a drink order -/
structure DrinkOrder where
  name : String
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of a drink order -/
def orderCost (order : DrinkOrder) : ℚ :=
  order.price * order.quantity

/-- Calculates the total cost of multiple drink orders -/
def totalCost (orders : List DrinkOrder) : ℚ :=
  orders.map orderCost |>.sum

/-- Calculates the change from a given amount -/
def calculateChange (paid : ℚ) (cost : ℚ) : ℚ :=
  paid - cost

theorem sandy_change_is_three :
  let orders : List DrinkOrder := [
    { name := "Cappuccino", price := 2, quantity := 3 },
    { name := "Iced Tea", price := 3, quantity := 2 },
    { name := "Cafe Latte", price := 1.5, quantity := 2 },
    { name := "Espresso", price := 1, quantity := 2 }
  ]
  let total := totalCost orders
  let paid := 20
  calculateChange paid total = 3 := by sorry

end sandy_change_is_three_l488_48852


namespace range_of_a_l488_48819

theorem range_of_a (x a : ℝ) :
  (∀ x, (x - 2) * (x - 3) < 0 → -4 < x - a ∧ x - a < 4) →
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end range_of_a_l488_48819


namespace birch_tree_probability_l488_48802

/-- The probability of no two birch trees being adjacent when planting trees in a row -/
theorem birch_tree_probability (maple oak birch : ℕ) (h1 : maple = 4) (h2 : oak = 3) (h3 : birch = 6) :
  let total := maple + oak + birch
  let non_birch := maple + oak
  let favorable := Nat.choose (non_birch + 1) birch
  let total_arrangements := Nat.choose total birch
  (favorable : ℚ) / total_arrangements = 7 / 429 :=
by sorry

end birch_tree_probability_l488_48802


namespace class_group_division_l488_48834

theorem class_group_division (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 32) (h2 : students_per_group = 6) :
  total_students / students_per_group = 5 :=
by sorry

end class_group_division_l488_48834


namespace sum_of_digits_499849_l488_48885

def number : Nat := 499849

def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem sum_of_digits_499849 :
  sumOfDigits number = 43 := by sorry

end sum_of_digits_499849_l488_48885


namespace inequality_proof_l488_48838

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_ineq : a + b < 2 * c) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end inequality_proof_l488_48838


namespace no_function_satisfies_inequality_l488_48848

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f x + f y ≥ 2 * f ((x + y) / 2) + 2 * |x - y| := by
  sorry

end no_function_satisfies_inequality_l488_48848


namespace max_min_sum_zero_l488_48883

def f (x : ℝ) := x^3 - 3*x

theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧
                (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧
                (m + n = 0) := by sorry

end max_min_sum_zero_l488_48883


namespace rohans_salary_l488_48841

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_expense_percent : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_expense_percent : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_expense_percent : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_expense_percent : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1500

/-- Theorem stating that Rohan's monthly salary is 7500 Rupees -/
theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent = 80 ∧
  savings = monthly_salary * (100 - (food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent)) / 100 :=
by sorry

end rohans_salary_l488_48841


namespace angle_line_plane_l488_48813

-- Define the line and plane
def line_eq1 (x z : ℝ) : Prop := x - 2*z + 3 = 0
def line_eq2 (y z : ℝ) : Prop := y + 3*z - 1 = 0
def plane_eq (x y z : ℝ) : Prop := 2*x - y + z + 3 = 0

-- Define the angle between the line and plane
def angle_between_line_and_plane : ℝ := sorry

-- State the theorem
theorem angle_line_plane :
  Real.sin angle_between_line_and_plane = 4 * Real.sqrt 21 / 21 :=
sorry

end angle_line_plane_l488_48813


namespace school_duration_in_minutes_l488_48899

-- Define the start and end times
def start_time : ℕ := 7
def end_time : ℕ := 11

-- Define the duration in hours
def duration_hours : ℕ := end_time - start_time

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem school_duration_in_minutes :
  duration_hours * minutes_per_hour = 240 :=
sorry

end school_duration_in_minutes_l488_48899


namespace fraction_simplification_l488_48842

theorem fraction_simplification : (3 : ℚ) / (2 - 2 / 5) = 15 / 8 := by
  sorry

end fraction_simplification_l488_48842


namespace divides_power_minus_constant_l488_48815

theorem divides_power_minus_constant (n : ℕ) : 13 ∣ 14^n - 27 := by
  sorry

end divides_power_minus_constant_l488_48815


namespace dog_speed_is_400_l488_48863

-- Define the constants from the problem
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def fetch_time_minutes : ℕ := 9
def feet_per_yard : ℕ := 3

-- Define the dog's speed as a function
def dog_speed : ℚ :=
  (football_fields * yards_per_field * feet_per_yard) / fetch_time_minutes

-- Theorem statement
theorem dog_speed_is_400 : dog_speed = 400 := by
  sorry

end dog_speed_is_400_l488_48863


namespace equal_count_for_any_number_l488_48879

/-- A function that represents the number of n-digit numbers from which 
    a k-digit number composed of only 1 and 2 can be obtained by erasing digits -/
def F (k n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is composed only of digits 1 and 2 -/
def OnlyOneTwo (x : ℕ) : Prop := sorry

theorem equal_count_for_any_number (k n : ℕ) (X Y : ℕ) (h1 : k > 0) (h2 : n ≥ k) 
  (hX : OnlyOneTwo X) (hY : OnlyOneTwo Y) 
  (hXdigits : X < 10^k) (hYdigits : Y < 10^k) : F k n = F k n := by
  sorry

end equal_count_for_any_number_l488_48879


namespace max_value_of_expression_l488_48880

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem max_value_of_expression (x y z : ℕ) 
  (h_two_digit_x : is_two_digit x)
  (h_two_digit_y : is_two_digit y)
  (h_two_digit_z : is_two_digit z)
  (h_mean : (x + y + z) / 3 = 60) :
  (∀ a b c : ℕ, is_two_digit a → is_two_digit b → is_two_digit c → 
    (a + b + c) / 3 = 60 → (a + b) / c ≤ 17) ∧
  (∃ a b c : ℕ, is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ 
    (a + b + c) / 3 = 60 ∧ (a + b) / c = 17) := by
  sorry

end max_value_of_expression_l488_48880


namespace carmen_initial_cats_l488_48809

/-- Represents the number of cats Carmen initially had -/
def initial_cats : ℕ := sorry

/-- Represents the number of dogs Carmen has -/
def dogs : ℕ := 18

/-- Represents the number of cats Carmen gave up for adoption -/
def cats_given_up : ℕ := 3

/-- Represents the difference between cats and dogs after giving up cats -/
def cat_dog_difference : ℕ := 7

theorem carmen_initial_cats :
  initial_cats = 28 ∧
  initial_cats - cats_given_up = dogs + cat_dog_difference :=
sorry

end carmen_initial_cats_l488_48809


namespace interest_calculation_l488_48846

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation (principal rate interest : ℝ) :
  principal = 26 →
  rate = 7 / 100 →
  interest = 10.92 →
  ∃ (time : ℝ), simple_interest principal rate time = interest ∧ time = 6 :=
by sorry

end interest_calculation_l488_48846


namespace speed_of_train2_l488_48807

-- Define the problem parameters
def distance_between_stations : ℝ := 200
def speed_of_train1 : ℝ := 20
def start_time_train1 : ℝ := 7
def start_time_train2 : ℝ := 8
def meeting_time : ℝ := 12

-- Define the theorem
theorem speed_of_train2 (speed_train2 : ℝ) : speed_train2 = 25 := by
  -- Assuming the conditions of the problem
  have h1 : distance_between_stations = 200 := by rfl
  have h2 : speed_of_train1 = 20 := by rfl
  have h3 : start_time_train1 = 7 := by rfl
  have h4 : start_time_train2 = 8 := by rfl
  have h5 : meeting_time = 12 := by rfl

  -- The proof would go here
  sorry

end speed_of_train2_l488_48807


namespace jason_egg_consumption_l488_48818

/-- The number of eggs Jason uses for one omelet -/
def eggs_per_omelet : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks_considered : ℕ := 2

/-- Theorem: Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_per_omelet * days_per_week * weeks_considered = 42 := by
  sorry

end jason_egg_consumption_l488_48818


namespace class_average_problem_l488_48894

theorem class_average_problem (n : ℝ) (h1 : n > 0) :
  let total_average : ℝ := 80
  let quarter_average : ℝ := 92
  let quarter_sum : ℝ := quarter_average * (n / 4)
  let total_sum : ℝ := total_average * n
  let rest_sum : ℝ := total_sum - quarter_sum
  let rest_average : ℝ := rest_sum / (3 * n / 4)
  rest_average = 76 := by
  sorry

end class_average_problem_l488_48894


namespace contractor_absent_days_l488_48851

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days given the contract details -/
def absentDays (c : ContractDetails) : ℚ :=
  (c.totalDays * c.paymentPerDay - c.totalReceived) / (c.paymentPerDay + c.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 10 -/
theorem contractor_absent_days :
  let c : ContractDetails := {
    totalDays := 30,
    paymentPerDay := 25,
    finePerDay := 15/2,
    totalReceived := 425
  }
  absentDays c = 10 := by sorry

end contractor_absent_days_l488_48851


namespace symmetrical_points_product_l488_48876

/-- 
Given two points P₁(a, 5) and P₂(-4, b) that are symmetrical about the x-axis,
prove that their x-coordinate product is -20.
-/
theorem symmetrical_points_product (a b : ℝ) : 
  (a = 4 ∧ b = -5) → a * b = -20 := by sorry

end symmetrical_points_product_l488_48876


namespace larger_number_problem_l488_48805

theorem larger_number_problem (x y : ℕ) : 
  x + y = 70 ∧ 
  y = 15 ∧ 
  x = 3 * y + 10 → 
  x = 55 := by
sorry

end larger_number_problem_l488_48805


namespace sum_distinct_prime_factors_of_7_power_difference_l488_48865

theorem sum_distinct_prime_factors_of_7_power_difference : 
  (Finset.sum (Finset.filter (Nat.Prime) (Finset.range ((7^7 - 7^4).factors.toFinset.card + 1)))
    (λ p => if p ∈ (7^7 - 7^4).factors.toFinset then p else 0)) = 31 := by sorry

end sum_distinct_prime_factors_of_7_power_difference_l488_48865


namespace expression_evaluation_l488_48860

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  b^2 - a^2 + 2*(a^2 + a*b) - (a^2 + b^2) = -1/3 :=
by sorry

end expression_evaluation_l488_48860


namespace rhombus_area_l488_48850

/-- A rhombus with side length √113 and diagonals differing by 10 units has area (√201)² - 5√201 -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 113 →
  d₂ = d₁ + 10 →
  d₁ * d₂ = 4 * s^2 →
  (1/2) * d₁ * d₂ = (Real.sqrt 201)^2 - 5 * Real.sqrt 201 := by
  sorry

end rhombus_area_l488_48850


namespace g_f_neg_four_equals_nine_l488_48823

/-- Given a function f and a function g, prove that g(f(-4)) = 9 
    under certain conditions. -/
theorem g_f_neg_four_equals_nine 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h1 : ∀ x, f x = 3 * x^2 - 7) 
  (h2 : g (f 4) = 9) : 
  g (f (-4)) = 9 := by
sorry

end g_f_neg_four_equals_nine_l488_48823


namespace like_terms_example_l488_48868

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → 
    (∃ (c1 c2 : ℚ), expr1 x y = c1 * x^5 * y^4 ∧ expr2 x y = c2 * x^5 * y^4)

theorem like_terms_example (a b : ℕ) (h1 : a = 2) (h2 : b = 3) :
  are_like_terms (λ x y => b * x^(2*a+1) * y^4) (λ x y => a * x^5 * y^(b+1)) :=
by
  sorry

end like_terms_example_l488_48868


namespace seating_uncertainty_l488_48830

-- Define the types for people and seats
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

inductive Seat : Type
| One : Seat
| Two : Seat
| Three : Seat
| Four : Seat

-- Define the seating arrangement
def Seating := Person → Seat

-- Define the "next to" relation
def next_to (s : Seating) (p1 p2 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three) ∨
  (s p1 = Seat.Three ∧ s p2 = Seat.Four) ∨
  (s p2 = Seat.One ∧ s p1 = Seat.Two) ∨
  (s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p2 = Seat.Three ∧ s p1 = Seat.Four)

-- Define the "between" relation
def between (s : Seating) (p1 p2 p3 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two ∧ s p3 = Seat.Three) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three ∧ s p3 = Seat.Four) ∨
  (s p3 = Seat.One ∧ s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p3 = Seat.Two ∧ s p2 = Seat.Three ∧ s p1 = Seat.Four)

theorem seating_uncertainty (s : Seating) :
  (next_to s Person.Dana Person.Bret) ∧
  (¬ between s Person.Abby Person.Bret Person.Carl) ∧
  (s Person.Bret = Seat.One) →
  ¬ (∀ p : Person, s p = Seat.Three → (p = Person.Abby ∨ p = Person.Carl)) :=
by sorry

end seating_uncertainty_l488_48830


namespace remaining_content_is_two_fifteenths_l488_48821

/-- The fraction of content remaining after four days of evaporation -/
def remaining_content : ℚ :=
  let day1_remaining := 1 - 2/3
  let day2_remaining := day1_remaining * (1 - 1/4)
  let day3_remaining := day2_remaining * (1 - 1/5)
  let day4_remaining := day3_remaining * (1 - 1/3)
  day4_remaining

/-- Theorem stating that the remaining content after four days is 2/15 -/
theorem remaining_content_is_two_fifteenths :
  remaining_content = 2/15 := by
  sorry

end remaining_content_is_two_fifteenths_l488_48821


namespace system_solution_l488_48816

theorem system_solution : 
  ∀ x y : ℝ, (x^3 + 3*x*y^2 = 49 ∧ x^2 + 8*x*y + y^2 = 8*y + 17*x) → 
  ((x = 1 ∧ y = 4) ∨ (x = 1 ∧ y = -4)) := by
sorry

end system_solution_l488_48816


namespace quadratic_equation_properties_l488_48889

/-- Given a quadratic equation mx^2 + nx - (m+n) = 0, prove that:
    1. The equation has two real roots.
    2. If n = 1 and the product of the roots is greater than 1, then -1/2 < m < 0. -/
theorem quadratic_equation_properties (m n : ℝ) :
  let f : ℝ → ℝ := λ x => m * x^2 + n * x - (m + n)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (n = 1 → (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ * x₂ > 1 → -1/2 < m ∧ m < 0)) :=
by sorry

end quadratic_equation_properties_l488_48889
