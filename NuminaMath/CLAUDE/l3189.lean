import Mathlib

namespace some_number_value_l3189_318901

theorem some_number_value (x : ℚ) :
  (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / x) - 1 / 16 = 0.24999999999999994 →
  x = 48 := by
  sorry

end some_number_value_l3189_318901


namespace prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l3189_318921

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Represents the day of the week -/
inductive Day
| Monday
| OtherDay

/-- Represents a child with their gender and birth day -/
structure Child :=
  (gender : Gender)
  (birthDay : Day)

/-- Represents a family with two children -/
structure Family :=
  (child1 : Child)
  (child2 : Child)

/-- The probability of having a boy or a girl is equal -/
axiom equal_gender_probability : ℝ

/-- The probability of being born on a Monday -/
axiom monday_probability : ℝ

/-- Theorem for the probability of having one boy and one girl in a family with two children -/
theorem prob_one_boy_one_girl : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy -/
theorem prob_one_boy_one_girl_given_boy : ℝ := by sorry

/-- Theorem for the probability of having one boy and one girl, given that one child is a boy born on a Monday -/
theorem prob_one_boy_one_girl_given_monday_boy : ℝ := by sorry

end prob_one_boy_one_girl_prob_one_boy_one_girl_given_boy_prob_one_boy_one_girl_given_monday_boy_l3189_318921


namespace tape_length_sum_l3189_318969

/-- Given three tapes A, B, and C with the following properties:
  * The length of tape A is 35 cm
  * The length of tape A is half the length of tape B
  * The length of tape C is 21 cm less than twice the length of tape A
  Prove that the sum of the lengths of tape B and tape C is 119 cm -/
theorem tape_length_sum (length_A length_B length_C : ℝ) : 
  length_A = 35 →
  length_A = length_B / 2 →
  length_C = 2 * length_A - 21 →
  length_B + length_C = 119 := by
  sorry

end tape_length_sum_l3189_318969


namespace equal_one_and_two_digit_prob_l3189_318913

def num_sides : ℕ := 15
def num_dice : ℕ := 5

def prob_one_digit : ℚ := 3 / 5
def prob_two_digit : ℚ := 2 / 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem equal_one_and_two_digit_prob :
  (choose num_dice (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) * (prob_one_digit ^ (num_dice / 2 + 1)) = 108 / 625 :=
sorry

end equal_one_and_two_digit_prob_l3189_318913


namespace tree_planting_cost_l3189_318994

/-- The cost to plant one tree given temperature drop and total cost -/
theorem tree_planting_cost 
  (temp_drop_per_tree : ℝ) 
  (total_temp_drop : ℝ) 
  (total_cost : ℝ) : 
  temp_drop_per_tree = 0.1 → 
  total_temp_drop = 1.8 → 
  total_cost = 108 → 
  (total_cost / (total_temp_drop / temp_drop_per_tree) = 6) :=
by
  sorry

#check tree_planting_cost

end tree_planting_cost_l3189_318994


namespace rectangle_area_l3189_318999

/-- Given a rectangle with perimeter 120 cm and length twice the width, prove its area is 800 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (2 * (length + width) = 120) →  -- Perimeter condition
  (length = 2 * width) →          -- Length-width relationship
  (length * width = 800) :=       -- Area to prove
by sorry

end rectangle_area_l3189_318999


namespace gcd_problem_l3189_318978

theorem gcd_problem (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end gcd_problem_l3189_318978


namespace cashew_price_satisfies_conditions_l3189_318936

/-- The price per pound of cashews that satisfies the mixture conditions -/
def cashew_price : ℝ := 6.75

/-- The total weight of the mixture in pounds -/
def total_mixture : ℝ := 50

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 5.70

/-- The weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- The price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5.00

/-- Theorem stating that the calculated cashew price satisfies the mixture conditions -/
theorem cashew_price_satisfies_conditions : 
  cashew_weight * cashew_price + (total_mixture - cashew_weight) * brazil_nut_price = 
  total_mixture * mixture_price :=
sorry

end cashew_price_satisfies_conditions_l3189_318936


namespace factorization_equality_l3189_318991

theorem factorization_equality (x : ℝ) : 4 * x - x^2 - 4 = -(x - 2)^2 := by
  sorry

end factorization_equality_l3189_318991


namespace circular_track_circumference_l3189_318976

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time : ℝ) 
  (h1 : speed1 = 7) 
  (h2 : speed2 = 8) 
  (h3 : meeting_time = 45) : 
  speed1 * meeting_time + speed2 * meeting_time = 675 := by
  sorry

end circular_track_circumference_l3189_318976


namespace number_plus_seven_equals_six_l3189_318951

theorem number_plus_seven_equals_six : 
  ∃ x : ℤ, x + 7 = 6 ∧ x = -1 := by
  sorry

end number_plus_seven_equals_six_l3189_318951


namespace largest_prime_factor_of_sum_of_divisors_180_l3189_318918

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 180 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 180 → q ≤ p ∧ p = 13 :=
sorry

end largest_prime_factor_of_sum_of_divisors_180_l3189_318918


namespace constant_term_expansion_l3189_318919

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), ∀ x, f x = (x + 1) * (x / 2 - a / Real.sqrt x)^6) →
  (∃ (g : ℝ → ℝ), ∀ x, g x = (x + 1) * (x / 2 - a / Real.sqrt x)^6 ∧ 
    (∃ c, c = 60 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |g x - c| < ε))) →
  a = 2 ∨ a = -2 :=
by sorry

end constant_term_expansion_l3189_318919


namespace binomial_variance_example_l3189_318943

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 3/4) is 3/2 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 3/4, by norm_num, by norm_num⟩
  variance X = 3/2 := by
  sorry

end binomial_variance_example_l3189_318943


namespace square_sum_simplification_l3189_318949

theorem square_sum_simplification : 99^2 + 202 * 99 + 101^2 = 40000 := by
  sorry

end square_sum_simplification_l3189_318949


namespace perimeter_of_modified_square_l3189_318958

/-- The perimeter of a figure ABFCDE formed by cutting a right triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) (triangle_leg : ℝ) 
  (h1 : side_length = 20)
  (h2 : triangle_leg = 12) : 
  let hypotenuse := Real.sqrt (2 * triangle_leg ^ 2)
  let perimeter := 2 * side_length + (side_length - triangle_leg) + hypotenuse + 2 * triangle_leg
  perimeter = 72 + 12 * Real.sqrt 2 := by
  sorry

end perimeter_of_modified_square_l3189_318958


namespace root_product_negative_l3189_318933

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem root_product_negative
  (f : ℝ → ℝ) (a x₁ x₂ : ℝ)
  (h_monotonic : Monotonic f)
  (h_root : f a = 0)
  (h_order : x₁ < a ∧ a < x₂) :
  f x₁ * f x₂ < 0 :=
sorry

end root_product_negative_l3189_318933


namespace proportional_survey_distribution_l3189_318917

/-- Represents the number of surveys to be drawn from a group -/
def surveyCount (totalSurveys : ℕ) (groupSize : ℕ) (totalPopulation : ℕ) : ℕ :=
  (totalSurveys * groupSize) / totalPopulation

theorem proportional_survey_distribution 
  (totalSurveys : ℕ) 
  (facultyStaffSize juniorHighSize seniorHighSize : ℕ) 
  (h1 : totalSurveys = 120)
  (h2 : facultyStaffSize = 500)
  (h3 : juniorHighSize = 3000)
  (h4 : seniorHighSize = 4000) :
  let totalPopulation := facultyStaffSize + juniorHighSize + seniorHighSize
  (surveyCount totalSurveys facultyStaffSize totalPopulation = 8) ∧ 
  (surveyCount totalSurveys juniorHighSize totalPopulation = 48) ∧
  (surveyCount totalSurveys seniorHighSize totalPopulation = 64) :=
by sorry

#check proportional_survey_distribution

end proportional_survey_distribution_l3189_318917


namespace value_of_expression_l3189_318984

theorem value_of_expression (x : ℝ) (h : x = 5) : 2 * x^2 + 3 = 53 := by
  sorry

end value_of_expression_l3189_318984


namespace pet_store_cages_l3189_318903

/-- Calculates the number of cages needed for a given number of animals and cage capacity -/
def cages_needed (animals : ℕ) (capacity : ℕ) : ℕ :=
  (animals + capacity - 1) / capacity

theorem pet_store_cages : 
  let initial_puppies : ℕ := 13
  let initial_kittens : ℕ := 10
  let initial_birds : ℕ := 15
  let sold_puppies : ℕ := 7
  let sold_kittens : ℕ := 4
  let sold_birds : ℕ := 5
  let puppy_capacity : ℕ := 2
  let kitten_capacity : ℕ := 3
  let bird_capacity : ℕ := 4
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_kittens := initial_kittens - sold_kittens
  let remaining_birds := initial_birds - sold_birds
  let total_cages := cages_needed remaining_puppies puppy_capacity + 
                     cages_needed remaining_kittens kitten_capacity + 
                     cages_needed remaining_birds bird_capacity
  total_cages = 8 := by
  sorry

end pet_store_cages_l3189_318903


namespace cassini_oval_properties_l3189_318927

-- Define the curve Γ
def Γ (m : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) * Real.sqrt ((x - 1)^2 + y^2) = m ∧ m > 0

-- Define a single-track curve
def SingleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, C x (f x)

-- Define a double-track curve
def DoubleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f g : ℝ → ℝ), (∀ x, C x (f x) ∨ C x (g x)) ∧ 
  (∃ x, f x ≠ g x)

-- The main theorem
theorem cassini_oval_properties :
  (∃ m : ℝ, m > 1 ∧ SingleTrackCurve (Γ m)) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ DoubleTrackCurve (Γ m)) := by
  sorry

end cassini_oval_properties_l3189_318927


namespace ice_volume_problem_l3189_318975

theorem ice_volume_problem (V : ℝ) : 
  (V * (1/4) * (1/4) = 0.4) → V = 6.4 := by
  sorry

end ice_volume_problem_l3189_318975


namespace longest_side_of_triangle_l3189_318946

/-- The longest side of a triangle with vertices at (1,1), (4,5), and (7,1) has a length of 6 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ), 
  a = (1, 1) ∧ b = (4, 5) ∧ c = (7, 1) ∧
  ∀ (d : ℝ), d = max (dist a b) (max (dist b c) (dist c a)) → d = 6 :=
by
  sorry

end longest_side_of_triangle_l3189_318946


namespace triangle_ratio_l3189_318920

theorem triangle_ratio (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  A = π/3 →  -- 60° in radians
  b = 1 → 
  S = Real.sqrt 3 → 
  S = (1/2) * b * c * Real.sin A → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  a / Real.sin A = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l3189_318920


namespace all_propositions_false_l3189_318970

-- Define a plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines in 3D space
def Line3D : Type := Set (ℝ × ℝ × ℝ)

-- Define the projection of a line onto a plane
def project (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Line3D := sorry

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define coincident lines
def coincide (l1 l2 : Line3D) : Prop := sorry

-- Define a line not on a plane
def not_on_plane (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem all_propositions_false (α : Set (ℝ × ℝ × ℝ)) :
  ∀ (m n : Line3D),
    not_on_plane m α → not_on_plane n α →
    (¬ (perpendicular (project m α) (project n α) → perpendicular m n)) ∧
    (¬ (perpendicular m n → perpendicular (project m α) (project n α))) ∧
    (¬ (intersect (project m α) (project n α) → intersect m n ∨ coincide m n)) ∧
    (¬ (parallel (project m α) (project n α) → parallel m n ∨ coincide m n)) :=
by sorry

end all_propositions_false_l3189_318970


namespace function_f_negative_two_l3189_318907

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- The main theorem -/
theorem function_f_negative_two (f : ℝ → ℝ) (h : FunctionF f) : f (-2) = 9 := by
  sorry

end function_f_negative_two_l3189_318907


namespace rhombus_equations_l3189_318981

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Point A of the rhombus -/
  A : ℝ × ℝ
  /-- Point C of the rhombus -/
  C : ℝ × ℝ
  /-- Point P on the line BC -/
  P : ℝ × ℝ
  /-- Assertion that ABCD is a rhombus -/
  is_rhombus : A = (-4, 7) ∧ C = (2, -3) ∧ P = (3, -1)

/-- The equation of line AD in a rhombus -/
def line_AD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 2 * x - y + 15 = 0

/-- The equation of diagonal BD in a rhombus -/
def diagonal_BD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 3 * x - 5 * y + 13 = 0

/-- Main theorem about the equations of line AD and diagonal BD in the given rhombus -/
theorem rhombus_equations (r : Rhombus) :
  (∀ x y, line_AD r x y ↔ y = 2 * x + 15) ∧
  (∀ x y, diagonal_BD r x y ↔ y = (3 * x + 13) / 5) := by
  sorry

end rhombus_equations_l3189_318981


namespace power_calculation_l3189_318987

theorem power_calculation : ((16^10 / 16^8)^3 * 8^3) / 2^9 = 16777216 := by sorry

end power_calculation_l3189_318987


namespace distance_after_5_hours_l3189_318952

/-- The distance between two people after walking in opposite directions for a given time -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two people walking in opposite directions for 5 hours,
    with speeds of 5 km/hr and 10 km/hr respectively, is 75 km -/
theorem distance_after_5_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end distance_after_5_hours_l3189_318952


namespace function_domain_implies_a_range_l3189_318940

/-- If the function f(x) = √(2^(x^2 + 2ax - a) - 1) is defined for all real x, then -1 ≤ a ≤ 0 -/
theorem function_domain_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^(x^2 + 2*a*x - a) - 1)) → 
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end function_domain_implies_a_range_l3189_318940


namespace certain_number_proof_l3189_318950

theorem certain_number_proof (p q : ℝ) (h1 : 3 / q = 18) (h2 : p - q = 7/12) : 3 / p = 4 := by
  sorry

end certain_number_proof_l3189_318950


namespace sunscreen_price_proof_l3189_318979

/-- Calculates the discounted price of sunscreen for a year --/
def discounted_sunscreen_price (bottles_per_month : ℕ) (months_per_year : ℕ) 
  (price_per_bottle : ℚ) (discount_percentage : ℚ) : ℚ :=
  let total_bottles := bottles_per_month * months_per_year
  let total_price := total_bottles * price_per_bottle
  let discount_amount := total_price * (discount_percentage / 100)
  total_price - discount_amount

/-- Proves that the discounted price of sunscreen for a year is $252.00 --/
theorem sunscreen_price_proof :
  discounted_sunscreen_price 1 12 30 30 = 252 := by
  sorry

#eval discounted_sunscreen_price 1 12 30 30

end sunscreen_price_proof_l3189_318979


namespace carmen_dogs_l3189_318980

def problem (initial_cats : ℕ) (adopted_cats : ℕ) (cat_dog_difference : ℕ) : Prop :=
  let remaining_cats := initial_cats - adopted_cats
  ∃ (dogs : ℕ), remaining_cats = dogs + cat_dog_difference

theorem carmen_dogs : 
  problem 28 3 7 → ∃ (dogs : ℕ), dogs = 18 :=
by
  sorry

end carmen_dogs_l3189_318980


namespace tom_dance_frequency_l3189_318968

/-- Represents the number of times Tom dances per week -/
def dance_frequency (hours_per_session : ℕ) (years : ℕ) (total_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (total_hours / (years * weeks_per_year)) / hours_per_session

/-- Proves that Tom dances 4 times a week given the conditions -/
theorem tom_dance_frequency :
  dance_frequency 2 10 4160 52 = 4 := by
sorry

end tom_dance_frequency_l3189_318968


namespace triangle_nth_root_l3189_318937

theorem triangle_nth_root (a b c : ℝ) (n : ℕ) (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) (h_n : n ≥ 2) :
  (a^(1/n) : ℝ) + (b^(1/n) : ℝ) > (c^(1/n) : ℝ) ∧
  (b^(1/n) : ℝ) + (c^(1/n) : ℝ) > (a^(1/n) : ℝ) ∧
  (a^(1/n) : ℝ) + (c^(1/n) : ℝ) > (b^(1/n) : ℝ) :=
by sorry

end triangle_nth_root_l3189_318937


namespace sum_of_four_powers_of_eight_l3189_318962

theorem sum_of_four_powers_of_eight :
  (8 : ℝ)^5 + (8 : ℝ)^5 + (8 : ℝ)^5 + (8 : ℝ)^5 = (8 : ℝ)^(17/3) := by
  sorry

end sum_of_four_powers_of_eight_l3189_318962


namespace matthew_crackers_l3189_318928

theorem matthew_crackers (initial_crackers : ℕ) 
  (friends : ℕ) 
  (crackers_eaten_per_friend : ℕ) 
  (crackers_left : ℕ) : 
  friends = 2 ∧ 
  crackers_eaten_per_friend = 6 ∧ 
  crackers_left = 11 ∧ 
  initial_crackers = friends * (crackers_eaten_per_friend * 2) + crackers_left → 
  initial_crackers = 35 := by
sorry

end matthew_crackers_l3189_318928


namespace f_properties_l3189_318982

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), 1 ≤ f x ∧ f x ≤ 2) ∧
  (f 0 = 1) ∧
  (f (Real.pi / 6) = 2) :=
by sorry

end f_properties_l3189_318982


namespace quadratic_inequality_solution_l3189_318939

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40*z + 350 ≤ 6 ↔ 20 - 2*Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2*Real.sqrt 14 := by
  sorry

end quadratic_inequality_solution_l3189_318939


namespace categorize_numbers_l3189_318961

def given_numbers : Set ℝ := {7/3, 1, 0, -1.4, Real.pi/2, 0.1010010001, -9}

def is_positive (x : ℝ) : Prop := x > 0

def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem categorize_numbers :
  let positive_numbers : Set ℝ := {7/3, 1, Real.pi/2, 0.1010010001}
  let fraction_numbers : Set ℝ := {7/3, -1.4, 0.1010010001}
  (∀ x ∈ given_numbers, is_positive x ↔ x ∈ positive_numbers) ∧
  (∀ x ∈ given_numbers, is_fraction x ↔ x ∈ fraction_numbers) := by
  sorry

end categorize_numbers_l3189_318961


namespace total_molecular_weight_theorem_l3189_318959

/-- Calculates the total molecular weight of given compounds -/
def totalMolecularWeight (Al_weight S_weight H_weight O_weight C_weight : ℝ) : ℝ :=
  let Al2S3_weight := 2 * Al_weight + 3 * S_weight
  let H2O_weight := 2 * H_weight + O_weight
  let CO2_weight := C_weight + 2 * O_weight
  7 * Al2S3_weight + 5 * H2O_weight + 4 * CO2_weight

/-- The total molecular weight of 7 moles of Al2S3, 5 moles of H2O, and 4 moles of CO2 is 1317.12 grams -/
theorem total_molecular_weight_theorem :
  totalMolecularWeight 26.98 32.06 1.01 16.00 12.01 = 1317.12 := by
  sorry

end total_molecular_weight_theorem_l3189_318959


namespace unknown_number_solution_l3189_318904

theorem unknown_number_solution :
  ∃! y : ℝ, (0.47 * 1442 - 0.36 * y) + 65 = 5 := by
  sorry

end unknown_number_solution_l3189_318904


namespace angle_between_given_lines_l3189_318941

def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0

def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1/3) := by sorry

end angle_between_given_lines_l3189_318941


namespace bank_coins_l3189_318947

theorem bank_coins (total_coins dimes quarters : ℕ) (h1 : total_coins = 11) (h2 : dimes = 2) (h3 : quarters = 7) :
  ∃ nickels : ℕ, nickels = total_coins - dimes - quarters :=
by
  sorry

end bank_coins_l3189_318947


namespace bicycle_cost_price_l3189_318900

/-- Proves that given a bicycle sold twice with profits of 20% and 25% respectively,
    and a final selling price of 225, the original cost price was 150. -/
theorem bicycle_cost_price 
  (profit_A : Real) 
  (profit_B : Real)
  (final_price : Real)
  (h1 : profit_A = 0.20)
  (h2 : profit_B = 0.25)
  (h3 : final_price = 225) :
  ∃ (initial_price : Real),
    initial_price * (1 + profit_A) * (1 + profit_B) = final_price ∧ 
    initial_price = 150 := by
  sorry

end bicycle_cost_price_l3189_318900


namespace range_of_a_l3189_318908

theorem range_of_a (a : ℝ) : Real.sqrt ((1 - 2*a)^2) = 2*a - 1 → a ≥ 1/2 := by
  sorry

end range_of_a_l3189_318908


namespace blue_cards_count_l3189_318960

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end blue_cards_count_l3189_318960


namespace unique_quadratic_root_l3189_318925

theorem unique_quadratic_root (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + k = 0) → k = 4 := by
  sorry

end unique_quadratic_root_l3189_318925


namespace percentage_increase_l3189_318964

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  (x - 70) / 70 * 100 = 11 := by
  sorry

end percentage_increase_l3189_318964


namespace probability_of_123456_l3189_318977

def num_cards : ℕ := 12
def num_distinct : ℕ := 6

def total_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => Nat.choose (num_cards - 2*i) 2))

def favorable_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => 2*i + 1))

theorem probability_of_123456 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 720 :=
sorry

end probability_of_123456_l3189_318977


namespace quadratic_roots_condition_l3189_318915

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) → 
  c = 9/5 := by
sorry

end quadratic_roots_condition_l3189_318915


namespace exponent_division_l3189_318965

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end exponent_division_l3189_318965


namespace expression_simplification_and_evaluation_l3189_318985

theorem expression_simplification_and_evaluation :
  let m : ℝ := Real.sqrt 3
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 :=
by sorry

end expression_simplification_and_evaluation_l3189_318985


namespace square_area_ratio_l3189_318986

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (s L : ℝ) (hs : s > 0) (hL : L > 0) 
    (h_perimeter : 4 * L = 4 * (4 * s)) : L^2 = 16 * s^2 := by
  sorry

end square_area_ratio_l3189_318986


namespace basketball_team_selection_l3189_318996

def total_players : ℕ := 12
def team_size : ℕ := 5
def captain_count : ℕ := 1
def regular_player_count : ℕ := 4

theorem basketball_team_selection :
  (total_players.choose captain_count) * ((total_players - captain_count).choose regular_player_count) = 3960 := by
  sorry

end basketball_team_selection_l3189_318996


namespace train_speed_problem_l3189_318953

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 500)
  (h2 : length2 = 750)
  (h3 : speed1 = 60)
  (h4 : time = 44.99640028797697) : 
  ∃ speed2 : ℝ, 
    speed2 = 40 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) :=
by sorry

end train_speed_problem_l3189_318953


namespace greatest_integer_less_than_negative_31_over_6_l3189_318966

theorem greatest_integer_less_than_negative_31_over_6 :
  ⌊-31/6⌋ = -6 := by sorry

end greatest_integer_less_than_negative_31_over_6_l3189_318966


namespace rectangular_room_tiles_l3189_318910

/-- Calculates the number of tiles touching the walls in a rectangular room -/
def tiles_touching_walls (length width : ℕ) : ℕ :=
  2 * length + 2 * width - 4

theorem rectangular_room_tiles (length width : ℕ) 
  (h_length : length = 10) (h_width : width = 5) : 
  tiles_touching_walls length width = 26 := by
  sorry

#eval tiles_touching_walls 10 5

end rectangular_room_tiles_l3189_318910


namespace range_of_a_l3189_318929

def p (a : ℝ) : Prop := 2 * a + 1 > 5
def q (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 3

theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 2) ∨ a > 3) :=
by sorry

end range_of_a_l3189_318929


namespace complex_on_line_l3189_318926

/-- Given a complex number z = (m-1) + (m+2)i that corresponds to a point on the line 2x-y=0,
    prove that m = 4. -/
theorem complex_on_line (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 1) (m + 2)
  2 * z.re - z.im = 0 → m = 4 := by
  sorry

end complex_on_line_l3189_318926


namespace angle_C_is_60_degrees_area_is_10_sqrt_3_l3189_318993

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - t.c) * (Real.sin t.A + Real.sin t.C) = (t.a - t.b) * Real.sin t.B

-- Theorem 1: Measure of angle C
theorem angle_C_is_60_degrees (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.C = Real.pi / 3 :=
sorry

-- Theorem 2: Area of triangle when a = 5 and c = 7
theorem area_is_10_sqrt_3 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t)
  (h3 : t.a = 5)
  (h4 : t.c = 7) : 
  (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3 :=
sorry

end angle_C_is_60_degrees_area_is_10_sqrt_3_l3189_318993


namespace babysitting_earnings_l3189_318909

/-- Represents the babysitting rates based on child's age --/
def BabysittingRate : ℕ → ℕ
  | age => if age < 2 then 5 else if age ≤ 5 then 7 else 8

/-- Calculates the total earnings from babysitting --/
def TotalEarnings (childrenAges : List ℕ) (hours : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ age hour => BabysittingRate age * hour) childrenAges hours)

theorem babysitting_earnings :
  let janeStartAge : ℕ := 18
  let childA : ℕ := janeStartAge / 2
  let childB : ℕ := childA - 2
  let childC : ℕ := childB + 3
  let childD : ℕ := childC
  let childrenAges : List ℕ := [childA, childB, childC, childD]
  let hours : List ℕ := [50, 90, 130, 70]
  TotalEarnings childrenAges hours = 2720 := by
  sorry


end babysitting_earnings_l3189_318909


namespace snail_noodles_problem_l3189_318988

/-- Snail noodles problem -/
theorem snail_noodles_problem 
  (price_A : ℝ) 
  (price_B : ℝ) 
  (quantity_A : ℝ) 
  (quantity_B : ℝ) 
  (h1 : price_A * quantity_A = 800)
  (h2 : price_B * quantity_B = 900)
  (h3 : price_B = 1.5 * price_A)
  (h4 : quantity_B = quantity_A - 2)
  (h5 : ∀ a : ℝ, 0 ≤ a ∧ a ≤ 15 → 
    90 * a + 135 * (30 - a) ≥ 90 * 15 + 135 * 15) :
  price_A = 100 ∧ price_B = 150 ∧ 
  (∃ (a : ℝ), 0 ≤ a ∧ a ≤ 15 ∧ 
    90 * a + 135 * (30 - a) = 3375 ∧
    ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 15 → 
      90 * b + 135 * (30 - b) ≥ 3375) :=
sorry

end snail_noodles_problem_l3189_318988


namespace combined_girls_average_is_85_l3189_318902

/-- Represents the average scores and student counts for two high schools -/
structure SchoolData where
  adams_boys_avg : ℝ
  adams_girls_avg : ℝ
  adams_combined_avg : ℝ
  baker_boys_avg : ℝ
  baker_girls_avg : ℝ
  baker_combined_avg : ℝ
  combined_boys_avg : ℝ
  adams_boys_count : ℝ
  adams_girls_count : ℝ
  baker_boys_count : ℝ
  baker_girls_count : ℝ

/-- Theorem stating that the combined girls' average score for both schools is 85 -/
theorem combined_girls_average_is_85 (data : SchoolData)
  (h1 : data.adams_boys_avg = 72)
  (h2 : data.adams_girls_avg = 78)
  (h3 : data.adams_combined_avg = 75)
  (h4 : data.baker_boys_avg = 84)
  (h5 : data.baker_girls_avg = 91)
  (h6 : data.baker_combined_avg = 85)
  (h7 : data.combined_boys_avg = 80)
  (h8 : data.adams_boys_count = data.adams_girls_count)
  (h9 : data.baker_boys_count = 6 * data.baker_girls_count / 7)
  (h10 : data.adams_boys_count = data.baker_boys_count) :
  (data.adams_girls_avg * data.adams_girls_count + data.baker_girls_avg * data.baker_girls_count) /
  (data.adams_girls_count + data.baker_girls_count) = 85 := by
  sorry


end combined_girls_average_is_85_l3189_318902


namespace stating_angle_bisector_division_l3189_318990

/-- Represents a parallelogram with sides of length 8 and 3 -/
structure Parallelogram where
  long_side : ℝ
  short_side : ℝ
  long_side_eq : long_side = 8
  short_side_eq : short_side = 3

/-- Represents the three parts of the divided side -/
structure DividedSide where
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ

/-- 
Theorem stating that the angle bisectors of the two angles adjacent to the longer side 
divide the opposite side into three parts with lengths 3, 2, and 3.
-/
theorem angle_bisector_division (p : Parallelogram) : 
  ∃ (d : DividedSide), d.part1 = 3 ∧ d.part2 = 2 ∧ d.part3 = 3 ∧ 
  d.part1 + d.part2 + d.part3 = p.long_side :=
sorry

end stating_angle_bisector_division_l3189_318990


namespace r_fourth_plus_inv_r_fourth_l3189_318972

theorem r_fourth_plus_inv_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inv_r_fourth_l3189_318972


namespace factor_t_squared_minus_64_l3189_318974

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l3189_318974


namespace children_count_l3189_318944

theorem children_count (pencils_per_child : ℕ) (skittles_per_child : ℕ) (total_pencils : ℕ) : 
  pencils_per_child = 2 → 
  skittles_per_child = 13 → 
  total_pencils = 18 → 
  total_pencils / pencils_per_child = 9 := by
sorry

end children_count_l3189_318944


namespace circle_intersections_l3189_318998

/-- A circle C with equation x^2 + y^2 - 2x - 4y - 4 = 0 -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- x₁ and x₂ are x-coordinates of intersection points with x-axis -/
def x_intersections (x₁ x₂ : ℝ) : Prop := C x₁ 0 ∧ C x₂ 0 ∧ x₁ ≠ x₂

/-- y₁ and y₂ are y-coordinates of intersection points with y-axis -/
def y_intersections (y₁ y₂ : ℝ) : Prop := C 0 y₁ ∧ C 0 y₂ ∧ y₁ ≠ y₂

theorem circle_intersections 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x_intersections x₁ x₂) 
  (hy : y_intersections y₁ y₂) : 
  abs (x₁ - x₂) = 2 * Real.sqrt 5 ∧ 
  y₁ + y₂ = 4 ∧ 
  x₁ * x₂ = y₁ * y₂ := by
  sorry

end circle_intersections_l3189_318998


namespace circle_center_l3189_318971

/-- The center of the circle defined by x^2 + y^2 - 4x - 2y - 5 = 0 is (2, 1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → (∃ r : ℝ, (x - 2)^2 + (y - 1)^2 = r^2) :=
by sorry

end circle_center_l3189_318971


namespace derivative_of_y_l3189_318995

noncomputable def y (x : ℝ) : ℝ := (Real.sin (x^2))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = 3 * x * Real.sin (x^2) * Real.sin (2 * x^2) :=
sorry

end derivative_of_y_l3189_318995


namespace vp_factorial_and_binomial_l3189_318924

/-- The p-adic valuation of a natural number -/
noncomputable def v_p (p : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of floor of N divided by increasing powers of p -/
def sum_floor (N : ℕ) (p : ℕ) : ℕ := sorry

theorem vp_factorial_and_binomial 
  (N k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_pow : ∃ n, N = p ^ n) (h_ge : N ≥ k) :
  (v_p p (N.factorial) = sum_floor N p) ∧ 
  (v_p p (Nat.choose N k) = v_p p N - v_p p k) := by
  sorry

end vp_factorial_and_binomial_l3189_318924


namespace same_color_plate_probability_l3189_318911

/-- The probability of selecting three plates of the same color from 7 green and 5 yellow plates. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 7 + 5
  let green_plates : ℕ := 7
  let yellow_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let green_combinations : ℕ := Nat.choose green_plates 3
  let yellow_combinations : ℕ := Nat.choose yellow_plates 3
  let same_color_combinations : ℕ := green_combinations + yellow_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry


end same_color_plate_probability_l3189_318911


namespace initial_investment_rate_l3189_318905

/-- Proves that the initial investment rate is 5% given the problem conditions --/
theorem initial_investment_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  (h5 : initial_investment + additional_investment = 12000) :
  ∃ R : ℝ, R = 5 ∧
    (initial_investment * R / 100 + additional_investment * additional_rate / 100 =
     (initial_investment + additional_investment) * total_rate / 100) :=
by
  sorry

end initial_investment_rate_l3189_318905


namespace triangle_side_length_l3189_318942

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))
  b = 2 * Real.sqrt 3 := by
sorry

end triangle_side_length_l3189_318942


namespace gcf_of_lcms_equals_15_l3189_318930

theorem gcf_of_lcms_equals_15 : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end gcf_of_lcms_equals_15_l3189_318930


namespace tshirts_per_package_l3189_318934

theorem tshirts_per_package (total_packages : ℕ) (total_tshirts : ℕ) 
  (h1 : total_packages = 71) 
  (h2 : total_tshirts = 426) : 
  total_tshirts / total_packages = 6 := by
  sorry

end tshirts_per_package_l3189_318934


namespace minimum_square_side_l3189_318932

theorem minimum_square_side (area_min : ℝ) (side : ℝ) : 
  area_min = 625 → side^2 ≥ area_min → side ≥ 0 → side ≥ 25 :=
by sorry

end minimum_square_side_l3189_318932


namespace weight_of_a_l3189_318957

theorem weight_of_a (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  ∃ e : ℝ, e = d + 5 ∧ (b + c + d + e) / 4 = 79 →
  a = 77 := by
sorry

end weight_of_a_l3189_318957


namespace quadratic_inequality_equivalent_to_interval_l3189_318935

theorem quadratic_inequality_equivalent_to_interval (x : ℝ) :
  x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_equivalent_to_interval_l3189_318935


namespace min_translation_for_symmetry_l3189_318967

/-- The minimum positive translation that makes the graph of a sine function symmetric about the origin -/
theorem min_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (x + π / 3 - φ)) →
  φ > 0 →
  (∀ x, f x = -f (-x)) →
  φ ≥ π / 3 ∧ 
  ∃ (φ_min : ℝ), φ_min = π / 3 ∧ 
    ∀ (ψ : ℝ), ψ > 0 → 
      (∀ x, 2 * Real.sin (x + π / 3 - ψ) = -(2 * Real.sin (-x + π / 3 - ψ))) → 
      ψ ≥ φ_min :=
by sorry


end min_translation_for_symmetry_l3189_318967


namespace rectangle_area_l3189_318973

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l3189_318973


namespace tate_high_school_duration_l3189_318914

theorem tate_high_school_duration (normal_hs_duration : ℕ) (total_time : ℕ) (x : ℕ) : 
  normal_hs_duration = 4 →
  total_time = 12 →
  (normal_hs_duration - x) + 3 * (normal_hs_duration - x) = total_time →
  x = 4 := by
sorry

end tate_high_school_duration_l3189_318914


namespace equation_solution_l3189_318912

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (7*x + 2) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2) ↔ 
  x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3 :=
by sorry

end equation_solution_l3189_318912


namespace A_3_2_equals_19_l3189_318923

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_19 : A 3 2 = 19 := by
  sorry

end A_3_2_equals_19_l3189_318923


namespace bridge_length_calculation_l3189_318956

/-- Given a train crossing a bridge and a lamp post, calculate the bridge length -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_crossing_time : ℝ) :
  train_length = 833.33 →
  bridge_crossing_time = 120 →
  lamp_post_crossing_time = 30 →
  ∃ bridge_length : ℝ, bridge_length = 2500 := by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l3189_318956


namespace first_nonzero_digit_after_decimal_of_1_157_l3189_318922

theorem first_nonzero_digit_after_decimal_of_1_157 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (1000 : ℚ) / 157 = 6 + (d : ℚ) / 10 + (n : ℚ) / 100 ∧ 
  d = 3 :=
sorry

end first_nonzero_digit_after_decimal_of_1_157_l3189_318922


namespace circle_trajectory_l3189_318963

/-- Given two circles and a line of symmetry, prove the trajectory of a third circle's center -/
theorem circle_trajectory (a l : ℝ) :
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 - a*x + 2*y + 1 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let symmetry_line := {(x, y) : ℝ × ℝ | y = x - l}
  let point_C := (-a, a)
  ∃ (center : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), center (x, y) ↔ 
      ((x + a)^2 + (y - a)^2 = x^2) ∧  -- P passes through C(-a, a) and is tangent to y-axis
      (∃ (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ circle1 ↔ 
        ∃ (q : ℝ × ℝ), q ∈ circle2 ∧ 
          (p.1 + q.1) / 2 = (p.2 + q.2) / 2 - l)) → -- symmetry condition
    (∀ (x y : ℝ), center (x, y) ↔ y^2 + 4*x - 4*y + 8 = 0) :=
by sorry

end circle_trajectory_l3189_318963


namespace factorization_equality_l3189_318954

theorem factorization_equality (m : ℝ) : m^2 * (m - 1) + 4 * (1 - m) = (m - 1) * (m + 2) * (m - 2) := by
  sorry

end factorization_equality_l3189_318954


namespace percentage_loss_calculation_l3189_318955

theorem percentage_loss_calculation (cost_price selling_price : ℝ) :
  cost_price = 1600 →
  selling_price = 1440 →
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end percentage_loss_calculation_l3189_318955


namespace main_theorem_l3189_318945

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0

/-- Proposition q -/
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

/-- Set A: values of x that satisfy p -/
def A : Set ℝ := {x | p x}

/-- Set B: values of x that satisfy q -/
def B (m : ℝ) : Set ℝ := {x | q x m}

/-- Main theorem: If ¬q is a sufficient but not necessary condition for ¬p,
    then m > 1 or m < -2 -/
theorem main_theorem (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, p x ∧ q x m) →
  m > 1 ∨ m < -2 :=
sorry

end main_theorem_l3189_318945


namespace right_triangles_bc_length_l3189_318916

/-- Given two right triangles ABC and ABD where B is vertically above A,
    and C and D lie on the horizontal axis, prove that if AC = 20, AD = 45,
    and BD = 13, then BC = 47. -/
theorem right_triangles_bc_length (A B C D : ℝ × ℝ) : 
  (∃ k : ℝ, B = (A.1, A.2 + k)) →  -- B is vertically above A
  (C.2 = A.2 ∧ D.2 = A.2) →        -- C and D lie on the horizontal axis
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- ABC is a right triangle
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 →  -- ABD is a right triangle
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 20 →  -- AC = 20
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 45 →  -- AD = 45
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 13 →  -- BD = 13
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 47    -- BC = 47
  := by sorry


end right_triangles_bc_length_l3189_318916


namespace window_area_ratio_l3189_318989

/-- Represents the window design with a rectangle and semicircles at each end -/
structure WindowDesign where
  /-- Total length of the window, including semicircles -/
  ad : ℝ
  /-- Diameter of the semicircles (width of the window) -/
  ab : ℝ
  /-- Ratio of total length to semicircle diameter is 4:3 -/
  ratio_condition : ad / ab = 4 / 3
  /-- The width of the window is 40 inches -/
  width_condition : ab = 40

/-- The ratio of the rectangle area to the semicircles area is 8/(3π) -/
theorem window_area_ratio (w : WindowDesign) :
  let r := w.ab / 2  -- radius of semicircles
  let rect_length := w.ad - w.ab  -- length of rectangle
  let rect_area := rect_length * w.ab  -- area of rectangle
  let semicircles_area := π * r^2  -- area of semicircles (full circle)
  rect_area / semicircles_area = 8 / (3 * π) := by
  sorry

end window_area_ratio_l3189_318989


namespace existence_of_point_S_l3189_318997

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is parallel to a plane -/
def is_parallel_to_plane (t : Triangle) (p : Plane) : Prop := sorry

/-- Finds the intersection point of a line and a plane -/
def line_plane_intersection (p1 p2 : Point3D) (plane : Plane) : Point3D := sorry

/-- The main theorem -/
theorem existence_of_point_S (α : Plane) (ABC MNP : Triangle) 
  (h : ¬ is_parallel_to_plane ABC α) : 
  ∃ (S : Point3D), 
    let A' := line_plane_intersection S ABC.A α
    let B' := line_plane_intersection S ABC.B α
    let C' := line_plane_intersection S ABC.C α
    let A'B'C' : Triangle := ⟨A', B', C'⟩
    are_congruent A'B'C' MNP := by
  sorry

end existence_of_point_S_l3189_318997


namespace grid_domino_coverage_l3189_318938

/-- Represents a 5x5 grid with a square removed at (i, j) -/
structure Grid :=
  (i : Nat) (j : Nat)

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Predicate to check if the grid can be covered by dominoes -/
def can_cover_with_dominoes (g : Grid) : Prop :=
  is_odd g.i ∧ is_odd g.j

theorem grid_domino_coverage (g : Grid) :
  (g.i ≤ 5 ∧ g.j ≤ 5) →
  (can_cover_with_dominoes g ↔ (is_odd g.i ∧ is_odd g.j)) :=
sorry

end grid_domino_coverage_l3189_318938


namespace sum_of_squares_l3189_318948

/-- Given a sequence {aₙ} where the sum of its first n terms S = 2n - 1,
    T is the sum of the first n terms of the sequence {aₙ²} -/
def T (n : ℕ) : ℚ :=
  (16^n - 1) / 15

/-- The sum of the first n terms of the original sequence -/
def S (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating that T is the correct sum for the sequence {aₙ²} -/
theorem sum_of_squares (n : ℕ) : T n = (16^n - 1) / 15 :=
  by sorry

end sum_of_squares_l3189_318948


namespace bbq_cooking_time_l3189_318931

/-- Calculates the time required to cook burgers for a BBQ --/
theorem bbq_cooking_time 
  (cooking_time_per_side : ℕ) 
  (grill_capacity : ℕ) 
  (total_guests : ℕ) 
  (guests_wanting_two : ℕ) 
  (guests_wanting_one : ℕ) 
  (h1 : cooking_time_per_side = 4)
  (h2 : grill_capacity = 5)
  (h3 : total_guests = 30)
  (h4 : guests_wanting_two = total_guests / 2)
  (h5 : guests_wanting_one = total_guests / 2)
  : (((guests_wanting_two * 2 + guests_wanting_one) / grill_capacity) * 
     (cooking_time_per_side * 2)) = 72 := by
  sorry

end bbq_cooking_time_l3189_318931


namespace intersection_of_sets_l3189_318906

open Set

theorem intersection_of_sets :
  let A : Set ℝ := {x | x > 2}
  let B : Set ℝ := {x | (x - 1) * (x - 3) < 0}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_sets_l3189_318906


namespace min_inequality_solution_l3189_318992

theorem min_inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z ≤ min (4 * (x - 1 / y)) (min (4 * (y - 1 / z)) (4 * (z - 1 / x)))) :
  x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2 := by
sorry

end min_inequality_solution_l3189_318992


namespace set_membership_solution_l3189_318983

theorem set_membership_solution (x : ℝ) :
  let A : Set ℝ := {2, x, x^2 + x}
  6 ∈ A → x = 6 ∨ x = -3 :=
by
  sorry

end set_membership_solution_l3189_318983
