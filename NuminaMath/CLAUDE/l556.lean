import Mathlib

namespace randy_house_blocks_l556_55666

/-- The number of blocks Randy used to build the house -/
def blocks_for_house : ℕ := 20

/-- The total number of blocks Randy has -/
def total_blocks : ℕ := 95

/-- The number of blocks Randy used to build the tower -/
def blocks_for_tower : ℕ := 50

theorem randy_house_blocks :
  blocks_for_house = 20 ∧
  total_blocks = 95 ∧
  blocks_for_tower = 50 ∧
  blocks_for_tower = blocks_for_house + 30 :=
sorry

end randy_house_blocks_l556_55666


namespace max_clerks_results_l556_55635

theorem max_clerks_results (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 100)
  (h2 : operation_count = initial_count - 1) :
  ∃ (max_results : ℕ), max_results = operation_count / 2 + 1 ∧ 
  max_results = 51 := by
  sorry

end max_clerks_results_l556_55635


namespace perfect_square_sum_l556_55617

theorem perfect_square_sum (a b : ℤ) : 
  (∃ x : ℤ, a^4 + (a+b)^4 + b^4 = x^2) ↔ a = 0 ∧ b = 0 := by
  sorry

end perfect_square_sum_l556_55617


namespace folded_paper_perimeter_ratio_l556_55613

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original_side : ℝ := 8
  let large_rect : Rectangle := { width := original_side, height := original_side / 2 }
  let small_rect : Rectangle := { width := original_side / 2, height := original_side / 2 }
  (perimeter small_rect) / (perimeter large_rect) = 2 / 3 := by
  sorry

end folded_paper_perimeter_ratio_l556_55613


namespace equation_roots_l556_55671

theorem equation_roots (c d : ℝ) : 
  (∀ x, (x + c) * (x + d) * (x - 5) / ((x + 4)^2) = 0 → x = -c ∨ x = -d ∨ x = 5) ∧
  (∀ x, x ≠ -4 → (x + c) * (x + d) * (x - 5) / ((x + 4)^2) ≠ 0) ∧
  (∀ x, (x + 2*c) * (x + 6) * (x + 9) / ((x + d) * (x - 5)) = 0 ↔ x = -4) →
  c = 1 ∧ d ≠ -6 ∧ d ≠ -9 ∧ 100 * c + d = 93 :=
by sorry

end equation_roots_l556_55671


namespace inequality_proof_l556_55621

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_ineq : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) :
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end inequality_proof_l556_55621


namespace division_by_fraction_twelve_divided_by_one_fourth_l556_55677

theorem division_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth :
  12 / (1 / 4) = 48 := by sorry

end division_by_fraction_twelve_divided_by_one_fourth_l556_55677


namespace fathers_age_l556_55672

/-- Proves that given the conditions, the father's age is 70 years. -/
theorem fathers_age (man_age : ℕ) (father_age : ℕ) : 
  man_age = (2 / 5 : ℚ) * father_age →
  man_age + 14 = (1 / 2 : ℚ) * (father_age + 14) →
  father_age = 70 :=
by sorry

end fathers_age_l556_55672


namespace julia_tag_game_l556_55647

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) 
  (h1 : monday_kids = 4) 
  (h2 : tuesday_kids = 14) : 
  monday_kids + tuesday_kids = 18 := by
  sorry

end julia_tag_game_l556_55647


namespace triangle_max_perimeter_l556_55610

theorem triangle_max_perimeter :
  ∀ a b : ℕ,
  b = 4 * a →
  (a + b > 16 ∧ a + 16 > b ∧ b + 16 > a) →
  a + b + 16 ≤ 41 :=
by sorry

end triangle_max_perimeter_l556_55610


namespace A_intersect_B_is_empty_l556_55611

def A : Set ℝ := {0, 1, 2}

def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) ≤ 0}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end A_intersect_B_is_empty_l556_55611


namespace race_result_l556_55615

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distanceTraveled (runner : Runner) (t : ℝ) : ℝ :=
  runner.speed * t

/-- The race problem setup -/
def raceProblem : Prop :=
  ∃ (A B : Runner),
    -- The race is 1000 meters long
    distanceTraveled A A.time = 1000 ∧
    -- A finishes the race in 115 seconds
    A.time = 115 ∧
    -- B finishes 10 seconds after A
    B.time = A.time + 10 ∧
    -- The distance by which A beats B is 80 meters
    1000 - distanceTraveled B A.time = 80

theorem race_result : raceProblem := by
  sorry

#check race_result

end race_result_l556_55615


namespace remainder_of_2_pow_1999_plus_1_mod_17_l556_55651

theorem remainder_of_2_pow_1999_plus_1_mod_17 :
  (2^1999 + 1) % 17 = 10 := by sorry

end remainder_of_2_pow_1999_plus_1_mod_17_l556_55651


namespace basketball_practice_time_l556_55693

theorem basketball_practice_time (school_day_practice : ℕ) : 
  (5 * school_day_practice + 2 * (2 * school_day_practice) = 135) → 
  school_day_practice = 15 := by
sorry

end basketball_practice_time_l556_55693


namespace blue_balloons_count_l556_55695

def total_balloons : ℕ := 200
def red_percentage : ℚ := 35 / 100
def green_percentage : ℚ := 25 / 100
def purple_percentage : ℚ := 15 / 100

theorem blue_balloons_count :
  (total_balloons : ℚ) * (1 - (red_percentage + green_percentage + purple_percentage)) = 50 := by
  sorry

end blue_balloons_count_l556_55695


namespace three_digit_subtraction_result_zero_l556_55686

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Subtracts the sum of digits from a number -/
def subtract_sum_of_digits (n : ℕ) : ℕ :=
  n - sum_of_digits n

/-- Applies the subtraction process n times -/
def apply_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => apply_n_times n (subtract_sum_of_digits x)

/-- The main theorem to be proved -/
theorem three_digit_subtraction_result_zero (x : ThreeDigitNumber) :
  apply_n_times 100 x.value = 0 :=
sorry

end three_digit_subtraction_result_zero_l556_55686


namespace tony_sand_and_water_problem_l556_55698

/-- Represents the problem of Tony filling his sandbox with sand and drinking water --/
theorem tony_sand_and_water_problem 
  (bucket_capacity : ℕ)
  (sandbox_depth sandbox_width sandbox_length : ℕ)
  (sand_weight_per_cubic_foot : ℕ)
  (water_per_session : ℕ)
  (water_bottle_volume : ℕ)
  (water_bottle_cost : ℕ)
  (initial_money : ℕ)
  (change_after_buying : ℕ)
  (h1 : bucket_capacity = 2)
  (h2 : sandbox_depth = 2)
  (h3 : sandbox_width = 4)
  (h4 : sandbox_length = 5)
  (h5 : sand_weight_per_cubic_foot = 3)
  (h6 : water_per_session = 3)
  (h7 : water_bottle_volume = 15)
  (h8 : water_bottle_cost = 2)
  (h9 : initial_money = 10)
  (h10 : change_after_buying = 4) :
  (sandbox_depth * sandbox_width * sandbox_length * sand_weight_per_cubic_foot) / bucket_capacity / 
  ((initial_money - change_after_buying) / water_bottle_cost * water_bottle_volume / water_per_session) = 4 :=
by sorry

end tony_sand_and_water_problem_l556_55698


namespace optimal_boat_combinations_l556_55608

/-- Represents a combination of large and small boats -/
structure BoatCombination where
  large_boats : Nat
  small_boats : Nat

/-- Checks if a boat combination is valid for the given number of people -/
def is_valid_combination (total_people : Nat) (large_capacity : Nat) (small_capacity : Nat) (combo : BoatCombination) : Prop :=
  combo.large_boats * large_capacity + combo.small_boats * small_capacity = total_people

theorem optimal_boat_combinations : 
  ∃ (combo1 combo2 : BoatCombination),
    combo1 ≠ combo2 ∧
    is_valid_combination 43 7 4 combo1 ∧
    is_valid_combination 43 7 4 combo2 :=
by sorry

end optimal_boat_combinations_l556_55608


namespace cubic_equation_roots_l556_55687

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c) ∧
    a + b + c = 9) →
  p + q = 38 := by
sorry

end cubic_equation_roots_l556_55687


namespace correct_group_sizes_l556_55690

/-- Represents the pricing structure and group information for a scenic area in Xi'an --/
structure ScenicAreaPricing where
  regularPrice : ℕ
  nonHolidayDiscount : ℚ
  holidayDiscountThreshold : ℕ
  holidayDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the cost for a group visiting on a non-holiday --/
def nonHolidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  pricing.regularPrice * (1 - pricing.nonHolidayDiscount) * people

/-- Calculates the cost for a group visiting on a holiday --/
def holidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  if people ≤ pricing.holidayDiscountThreshold then
    pricing.regularPrice * people
  else
    pricing.regularPrice * pricing.holidayDiscountThreshold +
    pricing.regularPrice * (1 - pricing.holidayDiscount) * (people - pricing.holidayDiscountThreshold)

/-- Theorem stating the correct number of people in each group --/
theorem correct_group_sizes (pricing : ScenicAreaPricing)
  (h1 : pricing.regularPrice = 50)
  (h2 : pricing.nonHolidayDiscount = 0.4)
  (h3 : pricing.holidayDiscountThreshold = 10)
  (h4 : pricing.holidayDiscount = 0.2)
  (h5 : pricing.totalPeople = 50)
  (h6 : pricing.totalCost = 1840) :
  ∃ (groupA groupB : ℕ),
    groupA + groupB = pricing.totalPeople ∧
    holidayCost pricing groupA + nonHolidayCost pricing groupB = pricing.totalCost ∧
    groupA = 24 ∧ groupB = 26 := by
  sorry


end correct_group_sizes_l556_55690


namespace integer_An_l556_55655

theorem integer_An (a b : ℕ+) (h1 : a > b) (θ : Real) 
  (h2 : 0 < θ) (h3 : θ < Real.pi / 2) 
  (h4 : Real.sin θ = (2 * a * b : ℝ) / ((a * a + b * b) : ℝ)) :
  ∀ n : ℕ, ∃ k : ℤ, (((a * a + b * b) : ℝ) ^ n * Real.sin (n * θ)) = k := by
  sorry

end integer_An_l556_55655


namespace lindas_painting_area_l556_55638

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening in a wall -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width * room.height + room.length * room.height)

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ :=
  rectangleArea opening.width opening.height

/-- Represents Linda's bedroom -/
def lindasBedroom : RoomDimensions := {
  width := 20,
  length := 20,
  height := 8
}

/-- Represents the doorway in Linda's bedroom -/
def doorway : Opening := {
  width := 3,
  height := 7
}

/-- Represents the window in Linda's bedroom -/
def window : Opening := {
  width := 6,
  height := 4
}

/-- Represents the closet doorway in Linda's bedroom -/
def closetDoorway : Opening := {
  width := 5,
  height := 7
}

/-- Theorem stating the total area of wall space Linda will have to paint -/
theorem lindas_painting_area :
  totalWallArea lindasBedroom -
  (openingArea doorway + openingArea window + openingArea closetDoorway) = 560 := by
  sorry

end lindas_painting_area_l556_55638


namespace kevins_cards_l556_55641

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) : 
  initial_cards = 7 → found_cards = 47 → initial_cards + found_cards = 54 := by
  sorry

end kevins_cards_l556_55641


namespace fraction_of_fraction_of_fraction_fraction_multiplication_l556_55673

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem fraction_multiplication :
  (1 / 5 : ℚ) * (1 / 3 : ℚ) * (1 / 4 : ℚ) * 120 = 2 := by sorry

end fraction_of_fraction_of_fraction_fraction_multiplication_l556_55673


namespace total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l556_55643

/- Define the trucks and their properties -/
structure Truck where
  initial_sand : Float
  sand_lost : Float
  miles_driven : Float

/- Define the four trucks -/
def truck1 : Truck := { initial_sand := 4.1, sand_lost := 2.4, miles_driven := 20 }
def truck2 : Truck := { initial_sand := 5.7, sand_lost := 3.6, miles_driven := 15 }
def truck3 : Truck := { initial_sand := 8.2, sand_lost := 1.9, miles_driven := 25 }
def truck4 : Truck := { initial_sand := 10.5, sand_lost := 2.1, miles_driven := 30 }

/- Calculate remaining sand for a truck -/
def remaining_sand (t : Truck) : Float :=
  t.initial_sand - t.sand_lost

/- Calculate percentage of sand lost for a truck -/
def percentage_lost (t : Truck) : Float :=
  (t.sand_lost / t.initial_sand) * 100

/- Theorem: Total remaining sand is 18.5 pounds -/
theorem total_remaining_sand :
  remaining_sand truck1 + remaining_sand truck2 + remaining_sand truck3 + remaining_sand truck4 = 18.5 := by
  sorry

/- Theorem: Percentage of sand lost by the first truck is 58.54% -/
theorem first_truck_percentage_lost :
  percentage_lost truck1 = 58.54 := by
  sorry

/- Theorem: Percentage of sand lost by the second truck is 63.16% -/
theorem second_truck_percentage_lost :
  percentage_lost truck2 = 63.16 := by
  sorry

/- Theorem: Percentage of sand lost by the third truck is 23.17% -/
theorem third_truck_percentage_lost :
  percentage_lost truck3 = 23.17 := by
  sorry

/- Theorem: Percentage of sand lost by the fourth truck is 20% -/
theorem fourth_truck_percentage_lost :
  percentage_lost truck4 = 20 := by
  sorry

end total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l556_55643


namespace combined_salaries_l556_55604

/-- The problem of calculating combined salaries -/
theorem combined_salaries 
  (salary_C : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_C = 11000)
  (h2 : average_salary = 8200)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_C = 30000 := by
  sorry

end combined_salaries_l556_55604


namespace square_difference_value_l556_55669

theorem square_difference_value (x y : ℝ) 
  (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : 
  (x - y)^2 = 9 := by
sorry

end square_difference_value_l556_55669


namespace gcd_problem_l556_55697

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 947 * k) : 
  Nat.gcd (Int.natAbs (3 * a^2 + 47 * a + 101)) (Int.natAbs (a + 19)) = 1 := by
  sorry

end gcd_problem_l556_55697


namespace CaCO3_molecular_weight_l556_55607

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Calcium atoms in CaCO3 -/
def num_Ca : ℕ := 1

/-- The number of Carbon atoms in CaCO3 -/
def num_C : ℕ := 1

/-- The number of Oxygen atoms in CaCO3 -/
def num_O : ℕ := 3

/-- The molecular weight of CaCO3 in g/mol -/
def molecular_weight_CaCO3 : ℝ :=
  num_Ca * atomic_weight_Ca + num_C * atomic_weight_C + num_O * atomic_weight_O

theorem CaCO3_molecular_weight :
  molecular_weight_CaCO3 = 100.09 := by sorry

end CaCO3_molecular_weight_l556_55607


namespace hyperbola_equation_l556_55694

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation (Γ : Set (ℝ × ℝ)) :
  (∃ F₁ F₂ : ℝ × ℝ, (∀ x y, ellipse x y ↔ (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)) ∧
                     (∀ x y, (x, y) ∈ Γ ↔ |(x - F₁.1)^2 + (y - F₁.2)^2 - (x - F₂.1)^2 - (y - F₂.2)^2| = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2))) →
  (∃ x y, (x, y) ∈ Γ ∧ asymptote x y) →
  ∃ x y, (x, y) ∈ Γ ↔ hyperbola_standard_form 27 9 x y :=
by sorry

end hyperbola_equation_l556_55694


namespace determine_sanity_with_one_question_l556_55631

-- Define the types
inductive Species : Type
| Human
| Vampire

inductive MentalState : Type
| Sane
| Insane

-- Define the Transylvanian type
structure Transylvanian :=
  (species : Species)
  (mental_state : MentalState)

-- Define the question type
inductive Question : Type
| AreYouAPerson

-- Define the answer type
inductive Answer : Type
| Yes
| No

-- Define the response function
def respond (t : Transylvanian) (q : Question) : Answer :=
  match t.mental_state, q with
  | MentalState.Sane, Question.AreYouAPerson => Answer.Yes
  | MentalState.Insane, Question.AreYouAPerson => Answer.No

-- Theorem statement
theorem determine_sanity_with_one_question :
  ∃ (q : Question), ∀ (t : Transylvanian),
    (respond t q = Answer.Yes ↔ t.mental_state = MentalState.Sane) ∧
    (respond t q = Answer.No ↔ t.mental_state = MentalState.Insane) :=
by sorry

end determine_sanity_with_one_question_l556_55631


namespace unique_prime_fraction_l556_55616

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_fraction :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    a ≠ b →
    is_prime (a * b^2 / (a + b)) →
    a = 6 ∧ b = 2 :=
by sorry

end unique_prime_fraction_l556_55616


namespace moon_speed_km_per_hour_l556_55670

/-- The speed of the moon around the earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.05

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * seconds_per_hour

/-- Theorem stating that the moon's speed in kilometers per hour is 3780 -/
theorem moon_speed_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3780 := by
  sorry

end moon_speed_km_per_hour_l556_55670


namespace intersection_with_complement_l556_55620

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 0}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end intersection_with_complement_l556_55620


namespace quadratic_factorization_l556_55618

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 24 * x^2 - 98 * x - 168 = (6 * x + a) * (4 * x + b)) →
  a + 2 * b = 10 := by
  sorry

end quadratic_factorization_l556_55618


namespace kimberly_skittles_l556_55696

/-- Calculates the total number of Skittles Kimberly has -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Proves that Kimberly has 12 Skittles in total -/
theorem kimberly_skittles : total_skittles 5 7 = 12 := by
  sorry

end kimberly_skittles_l556_55696


namespace copper_alloy_percentage_l556_55689

/-- Proves that the percentage of copper in the alloy that we need 32 kg of is 43.75% --/
theorem copper_alloy_percentage :
  ∀ (x : ℝ),
  -- Total mass of the final alloy
  let total_mass : ℝ := 40
  -- Percentage of copper in the final alloy
  let final_copper_percentage : ℝ := 45
  -- Mass of the alloy with unknown copper percentage
  let mass_unknown : ℝ := 32
  -- Mass of the alloy with 50% copper
  let mass_known : ℝ := 8
  -- Percentage of copper in the known alloy
  let known_copper_percentage : ℝ := 50
  -- The equation representing the mixture of alloys
  (mass_unknown * x / 100 + mass_known * known_copper_percentage / 100 = total_mass * final_copper_percentage / 100) →
  x = 43.75 := by
sorry

end copper_alloy_percentage_l556_55689


namespace volunteer_distribution_l556_55609

/-- The number of ways to distribute n distinguishable volunteers among k distinguishable places,
    such that each place has at least one volunteer. -/
def distribute_volunteers (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_distribution :
  distribute_volunteers 5 3 = 150 := by
  sorry

end volunteer_distribution_l556_55609


namespace odd_function_tangent_line_sum_l556_55678

def f (a b x : ℝ) : ℝ := a * x^3 + x + b

theorem odd_function_tangent_line_sum (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∃ m c : ℝ, ∀ x, m * x + c = f a b 1 + (3 * a * 1^2 + 1) * (x - 1) ∧ 
              m * 2 + c = 6) →  -- tangent line passes through (2, 6)
  a + b = 1 := by
sorry

end odd_function_tangent_line_sum_l556_55678


namespace expand_expression_l556_55600

theorem expand_expression (x y : ℝ) : (2*x + 15) * (3*y + 5) = 6*x*y + 10*x + 45*y + 75 := by
  sorry

end expand_expression_l556_55600


namespace f_derivative_at_2_l556_55633

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_derivative_at_2 : 
  deriv f 2 = (1 - Real.log 2) / 4 := by sorry

end f_derivative_at_2_l556_55633


namespace optimal_production_volume_l556_55619

-- Define the profit function
def W (x : ℝ) : ℝ := -2 * x^3 + 21 * x^2

-- State the theorem
theorem optimal_production_volume (x : ℝ) (h : x > 0) :
  ∃ (max_x : ℝ), max_x = 7 ∧ 
  ∀ y, y > 0 → W y ≤ W max_x :=
sorry

end optimal_production_volume_l556_55619


namespace quadratic_equation_properties_l556_55646

theorem quadratic_equation_properties (m : ℝ) :
  let f (x : ℝ) := x^2 - (2*m - 1)*x - 3*m^2 + m
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₂/x₁ + x₁/x₂ = -5/2 → (m = 1 ∨ m = 2/5)) :=
by sorry

end quadratic_equation_properties_l556_55646


namespace paco_cookies_l556_55675

/-- Given that Paco had 22 sweet cookies initially and ate 15 sweet cookies,
    prove that he had 7 sweet cookies left. -/
theorem paco_cookies (initial_sweet : ℕ) (eaten_sweet : ℕ) 
  (h1 : initial_sweet = 22) 
  (h2 : eaten_sweet = 15) : 
  initial_sweet - eaten_sweet = 7 := by
  sorry

end paco_cookies_l556_55675


namespace sqrt_inequality_l556_55684

theorem sqrt_inequality (n : ℝ) (h : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end sqrt_inequality_l556_55684


namespace min_value_sum_reciprocals_min_value_achieved_l556_55632

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

theorem min_value_achieved (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
  x' + y' + z' = 1 ∧ 
  (1 / (x' + y')) + ((x' + y') / z') = 3 := by
  sorry

end min_value_sum_reciprocals_min_value_achieved_l556_55632


namespace opposite_of_negative_fraction_l556_55650

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end opposite_of_negative_fraction_l556_55650


namespace students_either_not_both_is_38_l556_55676

/-- The number of students taking either geometry or history but not both -/
def students_either_not_both (students_both : ℕ) (students_geometry : ℕ) (students_only_history : ℕ) : ℕ :=
  (students_geometry - students_both) + students_only_history

/-- Theorem stating the number of students taking either geometry or history but not both -/
theorem students_either_not_both_is_38 :
  students_either_not_both 15 35 18 = 38 := by
  sorry

#check students_either_not_both_is_38

end students_either_not_both_is_38_l556_55676


namespace unique_number_satisfying_condition_l556_55622

theorem unique_number_satisfying_condition : ∃! x : ℕ, 143 - 10 * x = 3 * x := by
  sorry

end unique_number_satisfying_condition_l556_55622


namespace binary_multiplication_theorem_l556_55658

def binary_to_nat (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2 = 1) :: acc)
  to_binary_aux n []

def binary_mult (a b : List Bool) : List Bool :=
  nat_to_binary (binary_to_nat a * binary_to_nat b)

theorem binary_multiplication_theorem :
  binary_mult [true, false, false, true, true] [true, true, true] = 
  [true, true, true, true, true, false, true, false, true] := by
  sorry

end binary_multiplication_theorem_l556_55658


namespace simplify_absolute_value_expression_l556_55624

theorem simplify_absolute_value_expression
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y < 0)
  (hz : z < 0)
  (hxy : abs x > abs y)
  (hzx : abs z > abs x) :
  abs (x + z) - abs (y + z) - abs (x + y) = -2 * x :=
by sorry

end simplify_absolute_value_expression_l556_55624


namespace inequality_and_minimum_value_l556_55665

theorem inequality_and_minimum_value {a b x : ℝ} (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ y, y = (1 - x)^2 / x + x^2 / (1 - x) → y ≥ 1) :=
by sorry

end inequality_and_minimum_value_l556_55665


namespace all_defective_impossible_l556_55629

structure ProductSet where
  total : ℕ
  defective : ℕ
  drawn : ℕ
  h_total : total = 10
  h_defective : defective = 2
  h_drawn : drawn = 3
  h_defective_lt_total : defective < total

def all_defective (s : ProductSet) : Prop :=
  ∀ (i : Fin s.drawn), i.val < s.defective

theorem all_defective_impossible (s : ProductSet) : ¬ (all_defective s) := by
  sorry

end all_defective_impossible_l556_55629


namespace eight_amp_two_l556_55602

/-- Custom binary operation & -/
def amp (a b : ℤ) : ℤ := (a + b) * (a - b) + a * b

/-- Theorem: 8 & 2 = 76 -/
theorem eight_amp_two : amp 8 2 = 76 := by
  sorry

end eight_amp_two_l556_55602


namespace product_sum_zero_l556_55644

theorem product_sum_zero (a b c d : ℚ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) → 
  27*a + 9*b + 3*c + d = 0 := by
  sorry

end product_sum_zero_l556_55644


namespace purely_imaginary_reciprocal_l556_55664

theorem purely_imaginary_reciprocal (m : ℝ) :
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → z⁻¹ = Complex.I :=
by sorry

end purely_imaginary_reciprocal_l556_55664


namespace system_of_equations_solution_l556_55652

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 := by
  sorry

end system_of_equations_solution_l556_55652


namespace courier_net_pay_rate_l556_55614

def travel_time : ℝ := 3
def speed : ℝ := 65
def fuel_efficiency : ℝ := 28
def payment_rate : ℝ := 0.55
def gasoline_cost : ℝ := 2.50

theorem courier_net_pay_rate : 
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let earnings := payment_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_rate_per_hour := net_earnings / travel_time
  ⌊net_rate_per_hour⌋ = 30 := by sorry

end courier_net_pay_rate_l556_55614


namespace no_real_solutions_l556_55626

/-- 
Theorem: The system x^3 + y^3 = 2 and y = kx + d has no real solutions (x,y) 
if and only if k = -1 and 0 < d < 2√2.
-/
theorem no_real_solutions (k d : ℝ) : 
  (∀ x y : ℝ, x^3 + y^3 ≠ 2 ∨ y ≠ k*x + d) ↔ (k = -1 ∧ 0 < d ∧ d < 2 * Real.sqrt 2) := by
  sorry


end no_real_solutions_l556_55626


namespace simplify_sqrt_expression_l556_55642

theorem simplify_sqrt_expression :
  Real.sqrt 768 / Real.sqrt 192 - Real.sqrt 98 / Real.sqrt 49 = 2 - Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l556_55642


namespace meaningful_fraction_range_l556_55668

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 3) / (x - 2)) ↔ x ≥ -3 ∧ x ≠ 2 := by
  sorry

end meaningful_fraction_range_l556_55668


namespace simplify_complex_fraction_l556_55660

theorem simplify_complex_fraction :
  1 / ((3 / (Real.sqrt 5 + 2)) - (4 / (Real.sqrt 7 + 2))) =
  3 * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) /
  ((9 * Real.sqrt 5 - 4 * Real.sqrt 7 - 10) * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10)) :=
by sorry

end simplify_complex_fraction_l556_55660


namespace area_ratio_of_inscribed_squares_l556_55612

/-- A square inscribed in a circle -/
structure InscribedSquare where
  side : ℝ
  radius : ℝ
  inscribed : radius = side * Real.sqrt 2 / 2

/-- A square with two vertices on a line segment and two on a circle -/
structure PartiallyInscribedSquare where
  side : ℝ
  outer_square : InscribedSquare
  vertices_on_side : side ≤ outer_square.side
  vertices_on_circle : side = outer_square.side * Real.sqrt 2 / 5

/-- The theorem to be proved -/
theorem area_ratio_of_inscribed_squares (outer : InscribedSquare) 
    (inner : PartiallyInscribedSquare) (h : inner.outer_square = outer) :
    (inner.side ^ 2) / (outer.side ^ 2) = 2 / 25 := by
  sorry

end area_ratio_of_inscribed_squares_l556_55612


namespace correct_arrangement_l556_55601

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 6 ∧
  (∀ n, n ∈ arr → n ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ i, i < 3 → is_perfect_square (arr[2*i]! * arr[2*i+1]!))

theorem correct_arrangement :
  valid_arrangement [4, 2, 5, 3, 6, 1] :=
by sorry

end correct_arrangement_l556_55601


namespace even_function_implies_a_zero_l556_55682

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

end even_function_implies_a_zero_l556_55682


namespace max_red_squares_l556_55674

/-- A configuration of red squares on a 5x5 grid -/
def RedConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : RedConfiguration) : Nat :=
  sorry

/-- Checks if a configuration is valid (no axis-parallel rectangles) -/
def isValidConfiguration (config : RedConfiguration) : Prop :=
  ∀ p1 p2 p3 p4 : Fin 5 × Fin 5,
    config p1.1 p1.2 ∧ config p2.1 p2.2 ∧ config p3.1 p3.2 ∧ config p4.1 p4.2 →
    ¬isAxisParallelRectangle p1 p2 p3 p4

/-- The maximum number of red squares in a valid configuration is 12 -/
theorem max_red_squares :
  (∃ config : RedConfiguration, isValidConfiguration config ∧ countRedSquares config = 12) ∧
  (∀ config : RedConfiguration, isValidConfiguration config → countRedSquares config ≤ 12) :=
sorry

end max_red_squares_l556_55674


namespace ten_circles_l556_55627

/-- The maximum number of intersection points for n circles -/
def max_intersection_points (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * n

/-- Given conditions -/
axiom two_circles : max_intersection_points 2 = 2
axiom three_circles : max_intersection_points 3 = 6

/-- Theorem to prove -/
theorem ten_circles : max_intersection_points 10 = 90 := by
  sorry

end ten_circles_l556_55627


namespace find_m_l556_55634

theorem find_m : ∃ m : ℚ, m * 9999 = 624877405 ∧ m = 62493.5 := by
  sorry

end find_m_l556_55634


namespace sum_abc_equals_three_l556_55661

theorem sum_abc_equals_three (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 - 2*c = -1)
  (eq3 : c^2 - 6*a = -17) : 
  a + b + c = 3 := by
sorry

end sum_abc_equals_three_l556_55661


namespace rectangle_perimeter_l556_55688

/-- Given a square with perimeter 240 units divided into 4 congruent rectangles,
    where each rectangle's width is half the side length of the square,
    the perimeter of one rectangle is 180 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 240)
  (h2 : rectangle_count = 4) : ℝ :=
by
  -- Define the side length of the square
  let square_side := square_perimeter / 4

  -- Define the dimensions of each rectangle
  let rectangle_width := square_side / 2
  let rectangle_length := square_side

  -- Calculate the perimeter of one rectangle
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)

  -- Prove that the rectangle_perimeter equals 180
  sorry

#check rectangle_perimeter

end rectangle_perimeter_l556_55688


namespace amount_difference_l556_55663

theorem amount_difference (x : ℝ) (h : x = 690) : (0.25 * 1500) - (0.5 * x) = 30 := by
  sorry

end amount_difference_l556_55663


namespace smallest_solution_floor_equation_l556_55667

theorem smallest_solution_floor_equation : 
  ∀ x : ℝ, (x ≥ Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19) ∨ (x < Real.sqrt 119 ∧ ⌊x^2⌋ - ⌊x⌋^2 ≠ 19) :=
by sorry

end smallest_solution_floor_equation_l556_55667


namespace jordan_running_time_l556_55648

theorem jordan_running_time (steve_time steve_distance jordan_distance_1 jordan_distance_2 : ℚ)
  (h1 : steve_time = 32)
  (h2 : steve_distance = 4)
  (h3 : jordan_distance_1 = 3)
  (h4 : jordan_distance_2 = 7)
  (h5 : jordan_distance_1 / (steve_time / 2) = steve_distance / steve_time) :
  jordan_distance_2 / (jordan_distance_1 / (steve_time / 2)) = 112 / 3 := by sorry

end jordan_running_time_l556_55648


namespace quadratic_distinct_roots_l556_55636

/-- For a quadratic equation x^2 + 2x + 4c = 0 to have two distinct real roots, c must be less than 1/4 -/
theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end quadratic_distinct_roots_l556_55636


namespace rachelle_pennies_l556_55656

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ

/-- The conditions of the penny-throwing problem -/
def PennyConditions (p : PennyThrowers) : Prop :=
  p.gretchen = p.rachelle / 2 ∧
  p.rocky = p.gretchen / 3 ∧
  p.rachelle + p.gretchen + p.rocky = 300

/-- Theorem stating that under the given conditions, Rachelle threw 180 pennies -/
theorem rachelle_pennies (p : PennyThrowers) (h : PennyConditions p) : p.rachelle = 180 := by
  sorry

end rachelle_pennies_l556_55656


namespace michael_pizza_portion_l556_55605

theorem michael_pizza_portion
  (total_pizza : ℚ)
  (treshawn_portion : ℚ)
  (lamar_portion : ℚ)
  (h1 : total_pizza = 1)
  (h2 : treshawn_portion = 1 / 2)
  (h3 : lamar_portion = 1 / 6)
  : total_pizza - (treshawn_portion + lamar_portion) = 1 / 3 :=
by
  sorry

end michael_pizza_portion_l556_55605


namespace two_polygons_edges_l556_55649

theorem two_polygons_edges (a b : ℕ) : 
  a + b = 2014 →
  a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053 →
  a ≤ b →
  a = 952 := by sorry

end two_polygons_edges_l556_55649


namespace second_wave_infections_l556_55623

/-- Calculates the total number of infections over a given period, given an initial daily rate and an increase factor. -/
def totalInfections (initialRate : ℕ) (increaseFactor : ℕ) (days : ℕ) : ℕ :=
  (initialRate + initialRate * increaseFactor) * days

/-- Theorem stating that given an initial infection rate of 300 per day, 
    with a 4-fold increase in daily infections, the total number of 
    infections over a 14-day period is 21000. -/
theorem second_wave_infections : 
  totalInfections 300 4 14 = 21000 := by
  sorry

end second_wave_infections_l556_55623


namespace line_equation_through_parabola_intersection_l556_55606

/-- The equation of a line passing through (0, 2) and intersecting the parabola y² = 2x
    at two points M and N, where OM · ON = 0, is x + y - 2 = 0 -/
theorem line_equation_through_parabola_intersection (x y : ℝ) :
  let parabola := (fun (x y : ℝ) ↦ y^2 = 2*x)
  let line := (fun (x y : ℝ) ↦ ∃ (k : ℝ), y = k*x + 2)
  let O := (0, 0)
  let M := (x, y)
  let N := (2/y, y)  -- Using the parabola equation to express N
  parabola x y ∧
  line x y ∧
  (M.1 * N.1 + M.2 * N.2 = 0)  -- OM · ON = 0
  →
  x + y - 2 = 0 :=
by sorry

end line_equation_through_parabola_intersection_l556_55606


namespace line_point_at_t_4_l556_55685

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at : ℝ → ℝ × ℝ × ℝ

/-- Given a parameterized line with known points at t = 1 and t = -1, 
    prove that the point at t = 4 is (-27, 57, 27) -/
theorem line_point_at_t_4 
  (line : ParameterizedLine)
  (h1 : line.point_at 1 = (-3, 9, 12))
  (h2 : line.point_at (-1) = (4, -4, 2)) :
  line.point_at 4 = (-27, 57, 27) := by
sorry


end line_point_at_t_4_l556_55685


namespace solve_linear_system_l556_55637

theorem solve_linear_system (x y a : ℝ) 
  (eq1 : 4 * x + 3 * y = 1)
  (eq2 : a * x + (a - 1) * y = 3)
  (eq3 : x = y) :
  a = 11 := by
sorry

end solve_linear_system_l556_55637


namespace geometric_sequence_seventh_term_l556_55603

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence where the 4th term is 8 and the 10th term is 2, the 7th term is 1. -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_4th : a 4 = 8)
  (h_10th : a 10 = 2)
  : a 7 = 1 := by
  sorry

end geometric_sequence_seventh_term_l556_55603


namespace system_solution_existence_l556_55692

/-- The system of equations has at least one solution if and only if 0.5 ≤ a ≤ 2 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (|x - 0.5| + |y| - a) / (Real.sqrt 3 * y - x) = 0) ↔ 
  0.5 ≤ a ∧ a ≤ 2 :=
sorry

end system_solution_existence_l556_55692


namespace milkshake_cost_l556_55699

theorem milkshake_cost (initial_amount : ℕ) (hamburger_cost : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  remaining_amount = 70 →
  ∃ (milkshake_cost : ℕ), 
    initial_amount - (hamburger_cost * num_hamburgers) - (milkshake_cost * num_milkshakes) = remaining_amount ∧
    milkshake_cost = 3 :=
by sorry

end milkshake_cost_l556_55699


namespace third_side_is_seven_l556_55691

/-- A triangle with two known sides and even perimeter -/
structure EvenPerimeterTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 2
  side2_eq : side2 = 7
  even_perimeter : ∃ n : ℕ, side1 + side2 + side3 = 2 * n

/-- The third side of an EvenPerimeterTriangle is 7 -/
theorem third_side_is_seven (t : EvenPerimeterTriangle) : t.side3 = 7 := by
  sorry

end third_side_is_seven_l556_55691


namespace cookie_pie_leftover_slices_l556_55654

theorem cookie_pie_leftover_slices (num_pies : ℕ) (slices_per_pie : ℕ) (num_classmates : ℕ) (num_teachers : ℕ) (slices_per_person : ℕ) :
  num_pies = 3 →
  slices_per_pie = 10 →
  num_classmates = 24 →
  num_teachers = 1 →
  slices_per_person = 1 →
  num_pies * slices_per_pie - (num_classmates + num_teachers + 1) * slices_per_person = 4 :=
by sorry

end cookie_pie_leftover_slices_l556_55654


namespace circle_radius_l556_55659

/-- The equation of a circle in the form x^2 + y^2 + 2x = 0 has radius 1 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 + 2*x = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
  sorry

end circle_radius_l556_55659


namespace third_term_in_hundredth_group_l556_55657

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of terms before the nth group -/
def termsBeforeGroup (n : ℕ) : ℕ := triangularNumber (n - 1)

/-- The last term in the nth group -/
def lastTermInGroup (n : ℕ) : ℕ := sequenceTerm (termsBeforeGroup (n + 1))

/-- The kth term in the nth group -/
def termInGroup (n k : ℕ) : ℕ := lastTermInGroup n - 2 * (n - k)

theorem third_term_in_hundredth_group :
  termInGroup 100 3 = 9905 := by sorry

end third_term_in_hundredth_group_l556_55657


namespace citizenship_test_study_time_l556_55662

/-- Represents the time in minutes to learn each fill-in-the-blank question -/
def time_per_blank_question (total_questions : ℕ) (multiple_choice : ℕ) (fill_blank : ℕ) 
  (time_per_mc : ℕ) (total_study_time : ℕ) : ℕ :=
  ((total_study_time * 60) - (multiple_choice * time_per_mc)) / fill_blank

/-- Theorem stating that given the conditions, the time to learn each fill-in-the-blank question is 25 minutes -/
theorem citizenship_test_study_time :
  time_per_blank_question 60 30 30 15 20 = 25 := by
  sorry

end citizenship_test_study_time_l556_55662


namespace cubic_roots_sum_l556_55683

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2018*x + n = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
  sorry

end cubic_roots_sum_l556_55683


namespace evaluate_expression_l556_55639

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = -264 := by
  sorry

end evaluate_expression_l556_55639


namespace floor_negative_seven_fourths_l556_55680

theorem floor_negative_seven_fourths : ⌊(-7/4 : ℚ)⌋ = -2 := by sorry

end floor_negative_seven_fourths_l556_55680


namespace no_fourth_power_sum_1599_l556_55681

theorem no_fourth_power_sum_1599 :
  ¬ ∃ (s : Finset ℕ), (∀ n ∈ s, ∃ k, n = k^4) ∧ s.card ≤ 14 ∧ s.sum id = 1599 := by
  sorry

end no_fourth_power_sum_1599_l556_55681


namespace fuji_to_total_ratio_l556_55645

/-- Represents an apple orchard with Fuji and Gala trees -/
structure AppleOrchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The ratio of pure Fuji trees to all trees in the orchard is 39:52 -/
theorem fuji_to_total_ratio (orchard : AppleOrchard) :
  orchard.crossPollinated = (orchard.totalTrees : ℚ) * (1/10) ∧
  orchard.pureFuji + orchard.crossPollinated = 221 ∧
  orchard.pureGala = 39 →
  (orchard.pureFuji : ℚ) / orchard.totalTrees = 39 / 52 :=
by sorry

end fuji_to_total_ratio_l556_55645


namespace x_intercept_is_two_l556_55625

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 2 -/
theorem x_intercept_is_two :
  let l : Line := { x₁ := 1, y₁ := -2, x₂ := 5, y₂ := 6 }
  xIntercept l = 2 := by
  sorry

end x_intercept_is_two_l556_55625


namespace expected_lotus_seed_is_three_l556_55679

/-- The total number of zongzi -/
def total_zongzi : ℕ := 180

/-- The number of lotus seed zongzi -/
def lotus_seed_zongzi : ℕ := 54

/-- The size of the random sample -/
def sample_size : ℕ := 10

/-- The expected number of lotus seed zongzi in the sample -/
def expected_lotus_seed : ℚ := (sample_size : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ)

theorem expected_lotus_seed_is_three :
  expected_lotus_seed = 3 := by sorry

end expected_lotus_seed_is_three_l556_55679


namespace dot_product_theorem_l556_55653

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_theorem : (2 • a + b) • a = 1 := by sorry

end dot_product_theorem_l556_55653


namespace customer_b_bought_five_units_l556_55640

/-- Represents the phone inventory and sales of a store -/
structure PhoneStore where
  total_units : ℕ
  defective_units : ℕ
  customer_a_units : ℕ
  customer_c_units : ℕ

/-- Calculates the number of units sold to Customer B -/
def units_sold_to_b (store : PhoneStore) : ℕ :=
  store.total_units - store.defective_units - store.customer_a_units - store.customer_c_units

/-- Theorem stating that Customer B bought 5 units -/
theorem customer_b_bought_five_units (store : PhoneStore) 
  (h1 : store.total_units = 20)
  (h2 : store.defective_units = 5)
  (h3 : store.customer_a_units = 3)
  (h4 : store.customer_c_units = 7) :
  units_sold_to_b store = 5 := by
  sorry

end customer_b_bought_five_units_l556_55640


namespace handshake_partition_handshake_same_neighbors_l556_55628

open Set

structure HandshakeGraph (α : Type*) [Fintype α] where
  edges : Set (α × α)
  symm : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges
  handshake_property : ∀ a b c d, (a, b) ∈ edges → (b, c) ∈ edges → (c, d) ∈ edges →
    (a, c) ∈ edges ∨ (a, d) ∈ edges ∨ (b, d) ∈ edges

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem handshake_partition (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (X Y : Set (Fin n)), X.Nonempty ∧ Y.Nonempty ∧ X ∪ Y = univ ∧ X ∩ Y = ∅ ∧
  (∀ x y, x ∈ X → y ∈ Y → ((x, y) ∈ G.edges ↔ ∀ a ∈ X, ∀ b ∈ Y, (a, b) ∈ G.edges)) :=
sorry

theorem handshake_same_neighbors (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (A B : Fin n), A ≠ B ∧
  {x | x ≠ A ∧ x ≠ B ∧ (A, x) ∈ G.edges} = {x | x ≠ A ∧ x ≠ B ∧ (B, x) ∈ G.edges} :=
sorry

end handshake_partition_handshake_same_neighbors_l556_55628


namespace smallest_n_cookies_l556_55630

theorem smallest_n_cookies (n : ℕ) : (∃ k : ℕ, 15 * n - 1 = 11 * k) ↔ n ≥ 3 :=
  sorry

end smallest_n_cookies_l556_55630
