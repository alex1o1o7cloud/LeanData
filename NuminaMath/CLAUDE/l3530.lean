import Mathlib

namespace systematic_sampling_theorem_l3530_353092

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  studentsPerGroup : Nat
  selectedNumber : Nat
  selectedGroup : Nat

/-- Theorem: In a systematic sampling of 50 students into 10 groups of 5,
    if the student numbered 12 is selected from the third group,
    then the student numbered 37 will be selected from the eighth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.totalStudents = 50)
    (h2 : s.numGroups = 10)
    (h3 : s.studentsPerGroup = 5)
    (h4 : s.selectedNumber = 12)
    (h5 : s.selectedGroup = 3) :
    s.selectedNumber + (8 - s.selectedGroup) * s.studentsPerGroup = 37 := by
  sorry


end systematic_sampling_theorem_l3530_353092


namespace smallest_upper_bound_l3530_353079

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  ∀ y : ℝ, (∀ z : ℤ, (0 < z ∧ z < y ∧ 
                       0 < z ∧ z < 15 ∧
                       -1 < z ∧ z < 5 ∧
                       0 < z ∧ z < 3 ∧
                       z + 2 < 4 ∧
                       z = 1) → z ≤ x) → 
  y ≥ 2 :=
by sorry

end smallest_upper_bound_l3530_353079


namespace perpendicular_vector_solution_l3530_353064

def direction_vector : ℝ × ℝ := (2, 1)

theorem perpendicular_vector_solution :
  ∃! v : ℝ × ℝ, v.1 + v.2 = 1 ∧ v.1 * direction_vector.1 + v.2 * direction_vector.2 = 0 :=
by
  sorry

end perpendicular_vector_solution_l3530_353064


namespace bryan_pushups_l3530_353012

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduced_pushups : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduced_pushups)

/-- Theorem stating that Bryan did 100 push-ups in total -/
theorem bryan_pushups :
  total_pushups 9 12 8 = 100 := by
  sorry

end bryan_pushups_l3530_353012


namespace cycling_distance_is_four_point_five_l3530_353066

/-- Represents the cycling scenario with given conditions -/
structure CyclingScenario where
  speed : ℝ  -- Original speed in miles per hour
  time : ℝ   -- Original time taken in hours
  distance : ℝ -- Distance cycled in miles

/-- The conditions of the cycling problem -/
def cycling_conditions (scenario : CyclingScenario) : Prop :=
  -- Distance is speed multiplied by time
  scenario.distance = scenario.speed * scenario.time ∧
  -- Faster speed condition
  scenario.distance = (scenario.speed + 1/4) * (3/4 * scenario.time) ∧
  -- Slower speed condition
  scenario.distance = (scenario.speed - 1/4) * (scenario.time + 3)

/-- The theorem to be proved -/
theorem cycling_distance_is_four_point_five :
  ∀ (scenario : CyclingScenario), cycling_conditions scenario → scenario.distance = 4.5 := by
  sorry

end cycling_distance_is_four_point_five_l3530_353066


namespace pet_food_sale_discount_l3530_353076

def msrp : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def min_sale_price : ℝ := 25.20

theorem pet_food_sale_discount : ∃ (additional_discount : ℝ),
  additional_discount = 0.20 ∧
  min_sale_price = msrp * (1 - max_regular_discount) * (1 - additional_discount) :=
sorry

end pet_food_sale_discount_l3530_353076


namespace third_segment_length_l3530_353030

/-- Represents the lengths of interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Checks if the given segment lengths satisfy the radio show conditions. -/
def validSegments (s : InterviewSegments) : Prop :=
  s.first = 2 * (s.second + s.third) ∧
  s.third = s.second / 2 ∧
  s.first + s.second + s.third = 90

theorem third_segment_length :
  ∀ s : InterviewSegments, validSegments s → s.third = 10 := by
  sorry

end third_segment_length_l3530_353030


namespace total_profit_is_54000_l3530_353031

/-- Calculates the total profit given the investments and Jose's profit share -/
def calculate_total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- The total profit for Tom and Jose's business venture -/
theorem total_profit_is_54000 :
  calculate_total_profit 30000 12 45000 10 30000 = 54000 := by
  sorry

end total_profit_is_54000_l3530_353031


namespace proposition_a_proposition_d_l3530_353065

-- Proposition A
theorem proposition_a (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  -4 < a - b ∧ a - b < 2 := by sorry

-- Proposition D
theorem proposition_d : ∃ a : ℝ, a + 1 / a ≤ 2 := by sorry

end proposition_a_proposition_d_l3530_353065


namespace min_value_sum_reciprocals_l3530_353084

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n) + 1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_sum_reciprocals_l3530_353084


namespace entree_cost_l3530_353006

theorem entree_cost (total : ℝ) (difference : ℝ) (entree : ℝ) (dessert : ℝ)
  (h1 : total = 23)
  (h2 : difference = 5)
  (h3 : entree = dessert + difference)
  (h4 : total = entree + dessert) :
  entree = 14 := by
sorry

end entree_cost_l3530_353006


namespace possible_values_of_a_l3530_353024

def M : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end possible_values_of_a_l3530_353024


namespace f_above_x_axis_iff_valid_a_range_l3530_353051

/-- The function f(x) = (a^2 - 3a + 2)x^2 + (a - 1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 3*a + 2)*x^2 + (a - 1)*x + 2

/-- The graph of f(x) is above the x-axis -/
def above_x_axis (a : ℝ) : Prop := ∀ x, f a x > 0

/-- The range of values for a -/
def valid_a_range (a : ℝ) : Prop := a > 15/7 ∨ a ≤ 1

theorem f_above_x_axis_iff_valid_a_range :
  ∀ a : ℝ, above_x_axis a ↔ valid_a_range a := by sorry

end f_above_x_axis_iff_valid_a_range_l3530_353051


namespace george_second_day_hours_l3530_353044

/-- Calculates the hours worked on the second day given the hourly rate, 
    hours worked on the first day, and total earnings for two days. -/
def hoursWorkedSecondDay (hourlyRate : ℚ) (hoursFirstDay : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * hoursFirstDay) / hourlyRate

/-- Proves that given the specific conditions of the problem, 
    the hours worked on the second day is 2. -/
theorem george_second_day_hours : 
  hoursWorkedSecondDay 5 7 45 = 2 := by
  sorry

end george_second_day_hours_l3530_353044


namespace tau_fraction_values_l3530_353059

/-- The number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The number of positive divisors of n which have remainders 1 when divided by 3 -/
def τ₁ (n : ℕ+) : ℕ := sorry

/-- A number is composite if it's greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- The set of possible values for τ(10n) / τ₁(10n) -/
def possibleValues : Set ℕ := {n | n % 2 = 0 ∨ isComposite n}

/-- The main theorem -/
theorem tau_fraction_values (n : ℕ+) : 
  ∃ (k : ℕ), k ∈ possibleValues ∧ (τ (10 * n) : ℚ) / τ₁ (10 * n) = k := by sorry

end tau_fraction_values_l3530_353059


namespace rectangle_perimeter_l3530_353088

/-- Given a rectangle with length 8 and diagonal 17, its perimeter is 46 -/
theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 → 
  diagonal = 17 → 
  length^2 + width^2 = diagonal^2 → 
  2 * (length + width) = 46 :=
by
  sorry

end rectangle_perimeter_l3530_353088


namespace geometric_sequence_seventh_term_l3530_353022

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, 
    prove that the 7th term a₇ = 64. -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 2 + a 3 = 6) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) : 
  a 7 = 64 := by sorry

end geometric_sequence_seventh_term_l3530_353022


namespace brandy_caffeine_excess_l3530_353004

/-- Represents the caffeine consumption and limits for an individual -/
structure CaffeineProfile where
  weight : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℕ
  coffeinePer : ℝ
  energyDrinkConsumption : ℕ
  energyDrinkCaffeine : ℝ
  standardLimit : ℝ

/-- Calculates the total safe caffeine amount for an individual -/
def totalSafeAmount (profile : CaffeineProfile) : ℝ :=
  profile.weight * profile.standardLimit + profile.additionalTolerance

/-- Calculates the total caffeine consumed -/
def totalConsumed (profile : CaffeineProfile) : ℝ :=
  (profile.coffeeConsumption : ℝ) * profile.coffeinePer +
  (profile.energyDrinkConsumption : ℝ) * profile.energyDrinkCaffeine

/-- Theorem stating that Brandy has exceeded her safe caffeine limit by 470 mg -/
theorem brandy_caffeine_excess (brandy : CaffeineProfile)
  (h1 : brandy.weight = 60)
  (h2 : brandy.additionalTolerance = 50)
  (h3 : brandy.coffeeConsumption = 2)
  (h4 : brandy.coffeinePer = 95)
  (h5 : brandy.energyDrinkConsumption = 4)
  (h6 : brandy.energyDrinkCaffeine = 120)
  (h7 : brandy.standardLimit = 2.5) :
  totalConsumed brandy - totalSafeAmount brandy = 470 := by
  sorry

end brandy_caffeine_excess_l3530_353004


namespace point_on_x_axis_l3530_353069

/-- Given a point P with coordinates (3a-6, 1-a) that lies on the x-axis, 
    prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 3*a - 6 ∧ P.2 = 1 - a ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (-3, 0)) :=
by sorry

end point_on_x_axis_l3530_353069


namespace factor_expression_l3530_353023

theorem factor_expression (x : ℝ) : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) := by
  sorry

end factor_expression_l3530_353023


namespace roots_quadratic_equation_l3530_353057

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 2010 = 0) → 
  (b^2 + 3*b - 2010 = 0) → 
  (a^2 - a - 4*b = 2022) := by
  sorry

end roots_quadratic_equation_l3530_353057


namespace water_height_after_transfer_l3530_353070

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a rectangular tank given its dimensions and water height -/
def waterVolume (tank : TankDimensions) (waterHeight : ℝ) : ℝ :=
  tank.length * tank.width * waterHeight

/-- Theorem: The height of water in Tank A after transfer -/
theorem water_height_after_transfer (tankA : TankDimensions) (transferredVolume : ℝ) :
  tankA.length = 3 →
  tankA.width = 2 →
  tankA.height = 4 →
  transferredVolume = 12 →
  (waterVolume tankA (transferredVolume / (tankA.length * tankA.width))) = transferredVolume :=
by sorry

end water_height_after_transfer_l3530_353070


namespace x_squared_plus_k_factorization_l3530_353021

theorem x_squared_plus_k_factorization (k : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k = (x - a) * (x - b)) ↔ k = -1 := by
  sorry

end x_squared_plus_k_factorization_l3530_353021


namespace smallest_number_satisfying_conditions_l3530_353097

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, (21 ^ k ∣ n) → 7 ^ k - k ^ 7 = 1) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, (21 ^ k ∣ m) → 7 ^ k - k ^ 7 = 1) → m ≥ n) :=
by sorry

end smallest_number_satisfying_conditions_l3530_353097


namespace division_relation_l3530_353073

theorem division_relation : 
  (29.94 / 1.45 = 17.9) → (2994 / 14.5 = 1790) := by
  sorry

end division_relation_l3530_353073


namespace memory_card_cost_memory_card_cost_is_60_l3530_353010

/-- The cost of a single memory card given the following conditions:
  * John takes 10 pictures daily for 3 years
  * Each memory card stores 50 images
  * The total spent on memory cards is $13,140 -/
theorem memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (total_spent : ℕ) : ℕ :=
  let days_per_year : ℕ := 365
  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := total_pictures / images_per_card
  total_spent / cards_needed

/-- Proof that the cost of each memory card is $60 -/
theorem memory_card_cost_is_60 : memory_card_cost 10 3 50 13140 = 60 := by
  sorry

end memory_card_cost_memory_card_cost_is_60_l3530_353010


namespace sum_of_fractions_l3530_353058

theorem sum_of_fractions (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 3 := by
  sorry

end sum_of_fractions_l3530_353058


namespace area_of_specific_trapezoid_l3530_353008

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ
  /-- The center of the circle lies on the longer base of the trapezoid -/
  center_on_longer_base : Bool

/-- The area of an inscribed isosceles trapezoid -/
def area (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific inscribed trapezoid is 32 -/
theorem area_of_specific_trapezoid :
  let t : InscribedTrapezoid := ⟨5, 6, true⟩
  area t = 32 := by sorry

end area_of_specific_trapezoid_l3530_353008


namespace parallelogram_longer_side_length_l3530_353060

/-- Given a parallelogram with adjacent sides in the ratio 3:2 and perimeter 20,
    prove that the length of the longer side is 6. -/
theorem parallelogram_longer_side_length
  (a b : ℝ)
  (ratio : a / b = 3 / 2)
  (perimeter : 2 * (a + b) = 20)
  : a = 6 := by
  sorry

end parallelogram_longer_side_length_l3530_353060


namespace man_walked_five_minutes_l3530_353017

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  usual_travel_time : ℕ  -- Time it usually takes to drive from station to home
  early_arrival : ℕ      -- How early the man arrived at the station (in minutes)
  time_saved : ℕ         -- How much earlier they arrived home than usual

/-- Calculates the time the man spent walking given a WalkingScenario --/
def time_spent_walking (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific scenario, the man spent 5 minutes walking --/
theorem man_walked_five_minutes :
  let scenario : WalkingScenario := {
    usual_travel_time := 10,
    early_arrival := 60,
    time_saved := 10
  }
  time_spent_walking scenario = 5 := by
  sorry

end man_walked_five_minutes_l3530_353017


namespace joels_dads_age_l3530_353085

theorem joels_dads_age :
  ∀ (joel_current_age joel_future_age dads_current_age : ℕ),
    joel_current_age = 5 →
    joel_future_age = 27 →
    dads_current_age + (joel_future_age - joel_current_age) = 2 * joel_future_age →
    dads_current_age = 32 := by
  sorry

end joels_dads_age_l3530_353085


namespace coefficient_x3y5_proof_l3530_353094

/-- The coefficient of x^3y^5 in the expansion of (x+y)(x-y)^7 -/
def coefficient_x3y5 : ℤ := 14

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x3y5_proof :
  coefficient_x3y5 = (binomial 7 4 : ℤ) - (binomial 7 5 : ℤ) := by
  sorry

end coefficient_x3y5_proof_l3530_353094


namespace polynomial_factorization_l3530_353083

theorem polynomial_factorization (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 3) * (x - 2)) →
  (a = 1 ∧ b = -5 ∧ c = 6) := by
sorry

end polynomial_factorization_l3530_353083


namespace product_equality_l3530_353055

theorem product_equality : 50 * 29.96 * 2.996 * 500 = 2244004 := by
  sorry

end product_equality_l3530_353055


namespace no_negative_exponents_l3530_353086

theorem no_negative_exponents (a b c d : ℤ) (h : (5 : ℝ)^a + (5 : ℝ)^b = (2 : ℝ)^c + (2 : ℝ)^d + 17) :
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d :=
by sorry

end no_negative_exponents_l3530_353086


namespace two_number_problem_l3530_353096

theorem two_number_problem : ∃ x y : ℕ, 
  x ≠ y ∧ 
  x ≥ 10 ∧ 
  y ≥ 10 ∧ 
  (x + y) + (max x y - min x y) + (x * y) + (max x y / min x y) = 576 := by
  sorry

end two_number_problem_l3530_353096


namespace inverse_36_mod_53_l3530_353089

theorem inverse_36_mod_53 (h : (17⁻¹ : ZMod 53) = 26) : (36⁻¹ : ZMod 53) = 27 := by
  sorry

end inverse_36_mod_53_l3530_353089


namespace game_tie_fraction_l3530_353098

theorem game_tie_fraction (max_win_rate sara_win_rate postponed_rate : ℚ)
  (h_max : max_win_rate = 2/5)
  (h_sara : sara_win_rate = 1/4)
  (h_postponed : postponed_rate = 5/100) :
  let total_win_rate := max_win_rate + sara_win_rate
  let non_postponed_rate := 1 - postponed_rate
  let win_rate_of_non_postponed := total_win_rate / non_postponed_rate
  1 - win_rate_of_non_postponed = 6/19 := by
sorry

end game_tie_fraction_l3530_353098


namespace combine_terms_implies_zero_sum_l3530_353001

theorem combine_terms_implies_zero_sum (a b x y : ℝ) : 
  (∃ k : ℝ, -3 * a^(2*x-1) * b = k * 5 * a * b^(y+4)) → 
  (x - 2)^2016 + (y + 2)^2017 = 0 := by
sorry

end combine_terms_implies_zero_sum_l3530_353001


namespace exists_good_placement_l3530_353048

def RegularPolygon (n : ℕ) := Fin n → ℕ

def IsGoodPlacement (p : RegularPolygon 1983) : Prop :=
  ∀ (axis : Fin 1983), 
    ∀ (i : Fin 991), 
      p ((axis + i) % 1983) > p ((axis - i) % 1983)

theorem exists_good_placement : 
  ∃ (p : RegularPolygon 1983), IsGoodPlacement p :=
sorry

end exists_good_placement_l3530_353048


namespace max_blocks_fit_l3530_353016

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.height

/-- The dimensions of the larger box -/
def largeBox : BoxDimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of the smaller block -/
def smallBlock : BoxDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem: The maximum number of small blocks that can fit in the large box is 6 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 6 ∧ 
  (2 * smallBlock.length ≤ largeBox.length) ∧
  (smallBlock.width ≤ largeBox.width) ∧
  (2 * smallBlock.height ≤ largeBox.height) := by
  sorry

end max_blocks_fit_l3530_353016


namespace frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l3530_353011

theorem frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 2 / x > 1 ∧ x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(2 / x > 1)) :=
by
  sorry

end frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l3530_353011


namespace geometric_sequence_product_l3530_353005

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end geometric_sequence_product_l3530_353005


namespace mrs_hilt_remaining_cents_l3530_353029

/-- Given that Mrs. Hilt had 15 cents initially and spent 11 cents on a pencil, 
    prove that she was left with 4 cents. -/
theorem mrs_hilt_remaining_cents 
  (initial_cents : ℕ) 
  (pencil_cost : ℕ) 
  (h1 : initial_cents = 15)
  (h2 : pencil_cost = 11) :
  initial_cents - pencil_cost = 4 :=
by sorry

end mrs_hilt_remaining_cents_l3530_353029


namespace parabola_tangent_lines_l3530_353034

/-- A parabola defined by y^2 = 8x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- A line that has exactly one common point with the parabola and passes through P -/
def tangent_line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - P.2 = m * (p.1 - P.1)}

/-- The number of lines passing through P and having exactly one common point with the parabola -/
def num_tangent_lines : ℕ := 2

theorem parabola_tangent_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ m : ℝ, (tangent_line m ∩ parabola).Nonempty → m = m₁ ∨ m = m₂) ∧
  (tangent_line m₁ ∩ parabola).Nonempty ∧
  (tangent_line m₂ ∩ parabola).Nonempty :=
sorry

end parabola_tangent_lines_l3530_353034


namespace tan_half_sum_l3530_353074

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3) 
  (h2 : Real.sin p + Real.sin q = 8/17) : 
  Real.tan ((p + q) / 2) = 24/17 := by
  sorry

end tan_half_sum_l3530_353074


namespace paper_tearing_l3530_353095

theorem paper_tearing (n : ℕ) : 
  (∃ k : ℕ, 1 + 2 * k = 503) ∧ 
  (¬ ∃ k : ℕ, 1 + 2 * k = 2020) := by
  sorry

end paper_tearing_l3530_353095


namespace S_formula_l3530_353047

/-- g(k) is the largest odd factor of k -/
def g (k : ℕ+) : ℕ+ :=
  sorry

/-- Sn is the sum of g(k) for k from 1 to 2^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The main theorem: Sn = (1/3)(4^n + 2) for all natural numbers n -/
theorem S_formula (n : ℕ) : S n = (1/3) * (4^n + 2) :=
  sorry

end S_formula_l3530_353047


namespace total_albums_l3530_353035

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry

#check total_albums

end total_albums_l3530_353035


namespace percentage_increase_l3530_353091

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 80 → final = 120 → (final - initial) / initial * 100 = 50 := by
  sorry

end percentage_increase_l3530_353091


namespace prime_factorization_of_9600_l3530_353053

theorem prime_factorization_of_9600 : 9600 = 2^6 * 3 * 5^2 := by
  sorry

end prime_factorization_of_9600_l3530_353053


namespace workday_end_time_l3530_353049

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def workday_duration : Duration := ⟨8, 0⟩
def start_time : Time := ⟨7, 0, by sorry, by sorry⟩
def lunch_start : Time := ⟨11, 30, by sorry, by sorry⟩
def lunch_duration : Duration := ⟨0, 30⟩

/-- Adds a duration to a time -/
def add_duration (t : Time) (d : Duration) : Time :=
  sorry

/-- Subtracts two times to get a duration -/
def time_difference (t1 t2 : Time) : Duration :=
  sorry

theorem workday_end_time : 
  let time_before_lunch := time_difference lunch_start start_time
  let lunch_end := add_duration lunch_start lunch_duration
  let remaining_work := 
    ⟨workday_duration.hours - time_before_lunch.hours, 
     workday_duration.minutes - time_before_lunch.minutes⟩
  let end_time := add_duration lunch_end remaining_work
  end_time = ⟨15, 30, by sorry, by sorry⟩ := by
  sorry

end workday_end_time_l3530_353049


namespace gcd_factorial_eight_and_factorial_six_squared_l3530_353015

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l3530_353015


namespace saving_time_proof_l3530_353033

def down_payment : ℝ := 108000
def monthly_savings : ℝ := 3000
def months_in_year : ℝ := 12

theorem saving_time_proof : 
  (down_payment / monthly_savings) / months_in_year = 3 := by
sorry

end saving_time_proof_l3530_353033


namespace polar_midpoint_specific_case_l3530_353056

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (5, π/4) and (5, 3π/4) in polar coordinates is (5√2/2, π/2) -/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 5 (π/4) 5 (3*π/4)
  r = 5 * Real.sqrt 2 / 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end polar_midpoint_specific_case_l3530_353056


namespace sqrt_product_plus_factorial_equals_1114_l3530_353018

theorem sqrt_product_plus_factorial_equals_1114 : 
  Real.sqrt ((35 * 34 * 33 * 32) + 24) = 1114 := by
  sorry

end sqrt_product_plus_factorial_equals_1114_l3530_353018


namespace max_range_of_five_numbers_l3530_353003

theorem max_range_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- Distinct and ordered
  (a + b + c + d + e) / 5 = 13 →   -- Average is 13
  c = 15 →                         -- Median is 15
  e - a ≤ 33 :=                    -- Maximum range is at most 33
by sorry

end max_range_of_five_numbers_l3530_353003


namespace min_value_of_g_l3530_353068

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the property that f is decreasing on (-∞, 1]
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 1 → f a x ≥ f a y

-- Define the function g
def g (a : ℝ) : ℝ := f a (a + 1) - f a 1

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, is_decreasing_on_interval a → g a ≥ 1 ∧ ∃ a₀, g a₀ = 1 :=
sorry

end min_value_of_g_l3530_353068


namespace combination_equality_l3530_353043

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end combination_equality_l3530_353043


namespace zero_bounds_l3530_353045

theorem zero_bounds (a : ℝ) (x₀ : ℝ) (h_a : a > 0) 
  (h_zero : Real.exp (2 * x₀) + (a + 2) * Real.exp x₀ + a * x₀ = 0) : 
  Real.log (2 * a / (4 * a + 5)) < x₀ ∧ x₀ < -1 / Real.exp 1 := by
  sorry

end zero_bounds_l3530_353045


namespace pythagorean_triples_example_l3530_353042

-- Define a Pythagorean triple
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the two sets of triples
def triple1 : (ℕ × ℕ × ℕ) := (3, 4, 5)
def triple2 : (ℕ × ℕ × ℕ) := (6, 8, 10)

-- Theorem stating that both triples are Pythagorean triples
theorem pythagorean_triples_example :
  (is_pythagorean_triple triple1.1 triple1.2.1 triple1.2.2) ∧
  (is_pythagorean_triple triple2.1 triple2.2.1 triple2.2.2) :=
by sorry

end pythagorean_triples_example_l3530_353042


namespace rhombus_perimeter_l3530_353026

/-- Given a rhombus with area 24 and one diagonal of length 6, its perimeter is 20. -/
theorem rhombus_perimeter (area : ℝ) (diagonal1 : ℝ) (perimeter : ℝ) : 
  area = 24 → diagonal1 = 6 → perimeter = 20 := by
  sorry

end rhombus_perimeter_l3530_353026


namespace average_of_numbers_l3530_353052

def numbers : List ℝ := [3, 16, 33, 28]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 20 := by
  sorry

end average_of_numbers_l3530_353052


namespace find_a_l3530_353082

def U (a : ℝ) : Set ℝ := {3, a, a^2 + 2*a - 3}
def A : Set ℝ := {2, 3}

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, x ∈ U a → x ∈ A ∨ x = 5) ∧
  (∀ x : ℝ, x ∈ A → x ∈ U a) ∧
  a = 2 := by
sorry

end find_a_l3530_353082


namespace price_decrease_approx_16_67_percent_l3530_353019

/-- Calculates the percent decrease between two prices -/
def percentDecrease (oldPrice newPrice : ℚ) : ℚ :=
  (oldPrice - newPrice) / oldPrice * 100

/-- The original price per pack -/
def originalPricePerPack : ℚ := 9 / 6

/-- The promotional price per pack -/
def promotionalPricePerPack : ℚ := 10 / 8

/-- Theorem stating that the percent decrease in price per pack is approximately 16.67% -/
theorem price_decrease_approx_16_67_percent :
  abs (percentDecrease originalPricePerPack promotionalPricePerPack - 100 * (1 / 6)) < 1 / 100 := by
  sorry

end price_decrease_approx_16_67_percent_l3530_353019


namespace remainder_theorem_l3530_353009

theorem remainder_theorem : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by
  sorry

end remainder_theorem_l3530_353009


namespace ordered_pairs_satisfying_equation_l3530_353000

theorem ordered_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 83 = 24 * Nat.lcm a b + 17 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end ordered_pairs_satisfying_equation_l3530_353000


namespace smallest_a_is_54_l3530_353020

/-- A polynomial with three positive integer roots -/
structure PolynomialWithIntegerRoots where
  a : ℤ
  b : ℤ
  roots : Fin 3 → ℤ
  roots_positive : ∀ i, 0 < roots i
  polynomial_property : ∀ x, x^3 - a*x^2 + b*x - 30030 = (x - roots 0) * (x - roots 1) * (x - roots 2)

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℤ := 54

/-- Theorem stating that 54 is the smallest possible value of a -/
theorem smallest_a_is_54 (p : PolynomialWithIntegerRoots) : 
  p.a ≥ smallest_a ∧ ∃ (q : PolynomialWithIntegerRoots), q.a = smallest_a :=
sorry

end smallest_a_is_54_l3530_353020


namespace conditional_prob_one_jiuzhaigou_l3530_353072

/-- The number of attractions available for choice. -/
def num_attractions : ℕ := 5

/-- The probability that two people choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that exactly one person chooses Jiuzhaigou and they choose different attractions. -/
def prob_one_jiuzhaigou_different : ℚ := 8 / 25

/-- The conditional probability that exactly one person chooses Jiuzhaigou given that they choose different attractions. -/
theorem conditional_prob_one_jiuzhaigou (h : num_attractions = 5) :
  prob_one_jiuzhaigou_different / prob_different_attractions = 2 / 5 := by
  sorry

end conditional_prob_one_jiuzhaigou_l3530_353072


namespace quadratic_roots_range_l3530_353037

-- Define the quadratic function
def f (k x : ℝ) : ℝ := 2*k*x^2 - 2*x - 3*k - 2

-- Define the property of having two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Define the property of roots being on opposite sides of 1
def roots_around_one (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Theorem statement
theorem quadratic_roots_range (k : ℝ) :
  has_two_real_roots k ∧ roots_around_one k ↔ k < -4 ∨ k > 0 :=
sorry

end quadratic_roots_range_l3530_353037


namespace systematic_sampling_l3530_353087

theorem systematic_sampling 
  (total_students : Nat) 
  (num_segments : Nat) 
  (segment_size : Nat) 
  (sixteenth_segment_num : Nat) :
  total_students = 160 →
  num_segments = 20 →
  segment_size = 8 →
  sixteenth_segment_num = 125 →
  ∃ (first_segment_num : Nat),
    first_segment_num = 5 ∧
    sixteenth_segment_num = first_segment_num + segment_size * (16 - 1) :=
by sorry

end systematic_sampling_l3530_353087


namespace repeating_decimal_sum_l3530_353077

theorem repeating_decimal_sum : 
  let x : ℚ := 2 / 9
  let y : ℚ := 1 / 33
  x + y = 25 / 99 := by sorry

end repeating_decimal_sum_l3530_353077


namespace fifth_term_of_specific_geometric_sequence_l3530_353028

/-- Given a geometric sequence with first term a, common ratio r, and n-th term defined as a * r^(n-1) -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- The fifth term of a geometric sequence with first term 25 and common ratio -2 is 400 -/
theorem fifth_term_of_specific_geometric_sequence :
  let a := 25
  let r := -2
  geometric_sequence a r 5 = 400 := by
sorry

end fifth_term_of_specific_geometric_sequence_l3530_353028


namespace ratio_problem_l3530_353038

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  x / y = 13 / 2 := by
sorry

end ratio_problem_l3530_353038


namespace cylinder_height_in_hemisphere_l3530_353093

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  let h := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h = 2 * Real.sqrt 10 := by sorry

end cylinder_height_in_hemisphere_l3530_353093


namespace locus_of_circle_center_l3530_353025

/-
  Define the points M and N, and the circle passing through them with center P.
  Then prove that the locus of vertex P satisfies the given equation.
-/

-- Define the points M and N
def M : ℝ × ℝ := (0, -5)
def N : ℝ × ℝ := (0, 5)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  passes_through_M : (center.1 - M.1)^2 + (center.2 - M.2)^2 = (center.1 - N.1)^2 + (center.2 - N.2)^2

-- Define the locus equation
def locus_equation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.2^2 / 169 + P.1^2 / 144 = 1)

-- Theorem statement
theorem locus_of_circle_center (c : Circle) : locus_equation c.center :=
  sorry


end locus_of_circle_center_l3530_353025


namespace smallest_valid_number_l3530_353046

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  (n / 1000 ≠ n / 100 % 10) ∧ 
  (n / 1000 ≠ n / 10 % 10) ∧ 
  (n / 1000 ≠ n % 10) ∧ 
  (n / 100 % 10 ≠ n / 10 % 10) ∧ 
  (n / 100 % 10 ≠ n % 10) ∧ 
  (n / 10 % 10 ≠ n % 10) ∧  -- All digits are different
  (n / 1000 = 5 ∨ n / 100 % 10 = 5 ∨ n / 10 % 10 = 5 ∨ n % 10 = 5) ∧  -- Includes the digit 5
  (n % (n / 1000) = 0) ∧ 
  (n % (n / 100 % 10) = 0) ∧ 
  (n % (n / 10 % 10) = 0) ∧ 
  (n % (n % 10) = 0)  -- Divisible by each of its digits

theorem smallest_valid_number : 
  is_valid_number 5124 ∧ 
  ∀ m : ℕ, is_valid_number m → m ≥ 5124 :=
by sorry

end smallest_valid_number_l3530_353046


namespace quadratic_inequality_problem_l3530_353027

/-- Given that the solution set of ax^2 + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ 
   ∀ x : ℝ, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end quadratic_inequality_problem_l3530_353027


namespace multiple_of_nine_problem_l3530_353061

theorem multiple_of_nine_problem (N : ℕ) : 
  (∃ k : ℕ, N = 9 * k) →
  (∃ Q : ℕ, N = 9 * Q ∧ Q = 9 * 25 + 7) →
  N = 2088 := by
  sorry

end multiple_of_nine_problem_l3530_353061


namespace matrix_multiplication_result_l3530_353039

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 4, 5, 6; 7, 8, 9]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 1; 1, 1, 0; 0, 1, 1]
  A * B = !![3, 5, 4; 9, 11, 10; 15, 17, 16] := by sorry

end matrix_multiplication_result_l3530_353039


namespace batsman_average_increase_l3530_353054

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) (new_runs : Nat) (new_average : Rat) : Rat :=
  new_average - b.average

/-- Theorem: The increase in the batsman's average is 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
  b.innings = 16 →
  average_increase b 56 8 = 3 := by
  sorry


end batsman_average_increase_l3530_353054


namespace church_attendance_female_adults_l3530_353032

theorem church_attendance_female_adults
  (total : ℕ) (children : ℕ) (male_adults : ℕ)
  (h1 : total = 200)
  (h2 : children = 80)
  (h3 : male_adults = 60) :
  total - children - male_adults = 60 :=
by sorry

end church_attendance_female_adults_l3530_353032


namespace oven_temperature_l3530_353062

/-- Represents the temperature of the steak at time t -/
noncomputable def T (t : ℝ) : ℝ := sorry

/-- The constant oven temperature -/
def T_o : ℝ := sorry

/-- The initial temperature of the steak -/
def T_i : ℝ := 5

/-- The constant of proportionality in Newton's Law of Cooling -/
noncomputable def k : ℝ := sorry

/-- Newton's Law of Cooling: The rate of change of the steak's temperature
    is proportional to the difference between the steak's temperature and the oven temperature -/
axiom newtons_law_cooling : ∀ t, (deriv T t) = k * (T_o - T t)

/-- The solution to Newton's Law of Cooling -/
axiom cooling_solution : ∀ t, T t = T_o + (T_i - T_o) * Real.exp (-k * t)

/-- The temperature after 15 minutes is 45°C -/
axiom temp_at_15 : T 15 = 45

/-- The temperature after 30 minutes is 77°C -/
axiom temp_at_30 : T 30 = 77

/-- The theorem stating that the oven temperature is 205°C -/
theorem oven_temperature : T_o = 205 := by sorry

end oven_temperature_l3530_353062


namespace roots_equal_opposite_sign_l3530_353081

theorem roots_equal_opposite_sign (a b c d m : ℝ) : 
  (∀ x, (x^2 - 2*b*x + d) / (3*a*x - 4*c) = (m - 2) / (m + 2)) →
  (∃ r : ℝ, (r^2 - 2*b*r + d = 0) ∧ ((-r)^2 - 2*b*(-r) + d = 0)) →
  m = 4*b / (3*a - 2*b) :=
sorry

end roots_equal_opposite_sign_l3530_353081


namespace train_crossing_time_l3530_353067

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 160 ∧ 
  train_speed_kmh = 72 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3530_353067


namespace different_arrangements_count_l3530_353078

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements : ℕ := 15

theorem different_arrangements_count :
  (num_red_balls = 6) →
  (num_green_balls = 3) →
  (num_selected_balls = 4) →
  num_arrangements = 15 := by
  sorry

end different_arrangements_count_l3530_353078


namespace shifted_line_equation_l3530_353040

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift * l.slope }

/-- The original line y = x -/
def original_line : Line := { slope := 1, intercept := 0 }

theorem shifted_line_equation :
  let shifted := shift_line original_line (-1)
  shifted.slope = 1 ∧ shifted.intercept = 1 :=
sorry

end shifted_line_equation_l3530_353040


namespace range_of_a_l3530_353080

def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a (a : ℝ) : 
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end range_of_a_l3530_353080


namespace floor_plus_self_eq_fifteen_fourths_l3530_353090

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 15/4 :=
by sorry

end floor_plus_self_eq_fifteen_fourths_l3530_353090


namespace cylinder_volume_from_rectangle_l3530_353099

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 30) (h_width : width = 16) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 1920 * π := by sorry

end cylinder_volume_from_rectangle_l3530_353099


namespace cube_sum_product_l3530_353063

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by sorry

end cube_sum_product_l3530_353063


namespace chess_tournament_games_l3530_353036

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  n * (n - 1) = 90 → 2 * (n * (n - 1)) = 180 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l3530_353036


namespace max_xy_value_l3530_353014

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 := by
  sorry

end max_xy_value_l3530_353014


namespace minimum_value_reciprocal_sum_l3530_353075

theorem minimum_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end minimum_value_reciprocal_sum_l3530_353075


namespace min_value_trig_expression_l3530_353050

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 100 := by
  sorry

end min_value_trig_expression_l3530_353050


namespace right_triangle_leg_length_l3530_353041

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 4) 
  (hypotenuse : c = 5) : 
  b = 3 := by
sorry

end right_triangle_leg_length_l3530_353041


namespace value_of_A_minus_2B_A_minus_2B_independent_of_y_l3530_353013

/-- Definition of A in terms of x and y -/
def A (x y : ℝ) : ℝ := 2 * x^2 + x * y + 3 * y

/-- Definition of B in terms of x and y -/
def B (x y : ℝ) : ℝ := x^2 - x * y

/-- Theorem stating the value of A - 2B under the given condition -/
theorem value_of_A_minus_2B (x y : ℝ) :
  (x + 2)^2 + |y - 3| = 0 → A x y - 2 * B x y = -9 := by sorry

/-- Theorem stating the condition for A - 2B to be independent of y -/
theorem A_minus_2B_independent_of_y (x : ℝ) :
  (∀ y : ℝ, ∃ k : ℝ, A x y - 2 * B x y = k) ↔ x = -1 := by sorry

end value_of_A_minus_2B_A_minus_2B_independent_of_y_l3530_353013


namespace sum_of_largest_and_smallest_l3530_353071

/-- The set of available digits --/
def available_digits : Finset Nat := {0, 2, 4, 6}

/-- A function to check if a number is a valid three-digit number formed from the available digits --/
def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ available_digits ∧ b ∈ available_digits ∧ c ∈ available_digits ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 100 * a + 10 * b + c

/-- The largest valid number --/
def largest_number : Nat := 642

/-- The smallest valid number --/
def smallest_number : Nat := 204

/-- Theorem: The sum of the largest and smallest valid numbers is 846 --/
theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 846 := by
  sorry

end sum_of_largest_and_smallest_l3530_353071


namespace cos_72_degrees_l3530_353007

theorem cos_72_degrees : Real.cos (72 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end cos_72_degrees_l3530_353007


namespace repeating_decimal_to_fraction_l3530_353002

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), (x = 6 + (182 : ℚ) / 999) ∧ (1000 * x - x = 6182 - 6) :=
by
  sorry

end repeating_decimal_to_fraction_l3530_353002
