import Mathlib

namespace function_determination_l3964_396404

/-- Given two functions f and g with specific forms and conditions, prove they have specific expressions. -/
theorem function_determination (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ 2 * x^3 + a * x
  let g := fun (x : ℝ) ↦ b * x^2 + c
  (f 2 = 0) → 
  (g 2 = 0) → 
  (deriv f 2 = deriv g 2) →
  (f = fun (x : ℝ) ↦ 2 * x^3 - 8 * x) ∧ 
  (g = fun (x : ℝ) ↦ 4 * x^2 - 16) :=
by sorry

end function_determination_l3964_396404


namespace disco_vote_participants_l3964_396489

theorem disco_vote_participants :
  ∀ (initial_voters : ℕ) 
    (initial_oct22_percent initial_oct29_percent : ℚ)
    (additional_voters : ℕ)
    (final_oct29_percent : ℚ),
  initial_oct22_percent + initial_oct29_percent = 1 →
  initial_oct22_percent = 35 / 100 →
  initial_oct29_percent = 65 / 100 →
  additional_voters = 80 →
  final_oct29_percent = 45 / 100 →
  initial_oct29_percent * initial_voters = 
    final_oct29_percent * (initial_voters + additional_voters) →
  initial_voters + additional_voters = 260 := by
sorry


end disco_vote_participants_l3964_396489


namespace f_not_in_third_quadrant_l3964_396496

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating that the graph of f does not pass through the third quadrant -/
theorem f_not_in_third_quadrant :
  ∀ x : ℝ, ¬(in_third_quadrant x (f x)) :=
sorry

end f_not_in_third_quadrant_l3964_396496


namespace range_of_a_l3964_396400

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l3964_396400


namespace lesser_fraction_l3964_396439

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 10/11) (h_prod : x * y = 1/8) :
  min x y = (80 - 2 * Real.sqrt 632) / 176 := by sorry

end lesser_fraction_l3964_396439


namespace system_one_solution_system_two_solution_l3964_396458

-- System 1
theorem system_one_solution :
  ∃ (x y : ℚ), 3 * x + 2 * y = 8 ∧ y = 2 * x - 3 ∧ x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℚ), 2 * x + 3 * y = 6 ∧ 3 * x - 2 * y = -2 ∧ x = 6/13 ∧ y = 22/13 := by sorry

end system_one_solution_system_two_solution_l3964_396458


namespace negation_equivalence_l3964_396474

-- Define a type for polyhedra
structure Polyhedron where
  faces : Set Face

-- Define a type for faces
inductive Face
  | Triangle
  | Quadrilateral
  | Pentagon
  | Other

-- Define the original proposition
def original_proposition : Prop :=
  ∀ p : Polyhedron, ∃ f ∈ p.faces, f = Face.Triangle ∨ f = Face.Quadrilateral ∨ f = Face.Pentagon

-- Define the negation
def negation : Prop :=
  ∃ p : Polyhedron, ∀ f ∈ p.faces, f ≠ Face.Triangle ∧ f ≠ Face.Quadrilateral ∧ f ≠ Face.Pentagon

-- Theorem stating the equivalence
theorem negation_equivalence : ¬original_proposition ↔ negation := by
  sorry

end negation_equivalence_l3964_396474


namespace integer_part_sqrt_39_minus_3_l3964_396492

theorem integer_part_sqrt_39_minus_3 : 
  ⌊Real.sqrt 39 - 3⌋ = 3 := by sorry

end integer_part_sqrt_39_minus_3_l3964_396492


namespace correlation_coefficient_properties_l3964_396435

/-- The correlation coefficient between two variables -/
def correlation_coefficient (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

/-- The strength of correlation between two variables -/
def correlation_strength (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

theorem correlation_coefficient_properties (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] :
  let r := correlation_coefficient X Y
  ∃ (strength : ℝ → ℝ),
    (∀ x, |x| ≤ 1 → strength x ≥ 0) ∧
    (∀ x y, |x| ≤ 1 → |y| ≤ 1 → |x| < |y| → strength x < strength y) ∧
    (∀ x, |x| ≤ 1 → strength x = correlation_strength X Y) ∧
    |r| ≤ 1 :=
by sorry

end correlation_coefficient_properties_l3964_396435


namespace largest_three_digit_multiple_of_4_and_5_l3964_396449

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 4 ∣ n ∧ 5 ∣ n → n ≤ 980 :=
by
  sorry

#check largest_three_digit_multiple_of_4_and_5

end largest_three_digit_multiple_of_4_and_5_l3964_396449


namespace repeating_decimal_sum_l3964_396455

/-- Definition of a repeating decimal with a single digit repeating -/
def repeating_decimal (d : ℕ) : ℚ := (d : ℚ) / 9

/-- The problem statement -/
theorem repeating_decimal_sum : 
  repeating_decimal 6 + repeating_decimal 2 - repeating_decimal 4 = 4/9 := by
  sorry

end repeating_decimal_sum_l3964_396455


namespace initial_seashells_count_l3964_396452

/-- The number of seashells Jason found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Jason found -/
def starfish : ℕ := 48

/-- The number of seashells Jason gave to Tim -/
def seashells_given_away : ℕ := 13

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- Theorem stating that the initial number of seashells is equal to the current number plus the number given away -/
theorem initial_seashells_count : initial_seashells = current_seashells + seashells_given_away := by
  sorry

end initial_seashells_count_l3964_396452


namespace decagon_diagonals_l3964_396405

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals : 
  num_diagonals 4 = 2 ∧ num_diagonals 5 = 5 → num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l3964_396405


namespace factor_polynomial_l3964_396429

theorem factor_polynomial (x : ℝ) : 72 * x^7 - 250 * x^13 = 2 * x^7 * (2^2 * 3^2 - 5^3 * x^6) := by
  sorry

end factor_polynomial_l3964_396429


namespace mao_saying_moral_l3964_396401

/-- Represents the moral of a saying -/
inductive Moral
| KnowledgeDrivesPractice
| KnowledgeGuidesPractice
| PracticeSourceOfKnowledge
| PracticeSocialHistorical

/-- Represents a philosophical saying -/
structure Saying :=
(content : String)
(moral : Moral)

/-- Mao Zedong's saying about tasting a pear -/
def maoSaying : Saying :=
{ content := "If you want to know the taste of a pear, you must change the pear and taste it yourself",
  moral := Moral.PracticeSourceOfKnowledge }

/-- Theorem stating that the moral of Mao's saying is "Practice is the source of knowledge" -/
theorem mao_saying_moral :
  maoSaying.moral = Moral.PracticeSourceOfKnowledge :=
sorry

end mao_saying_moral_l3964_396401


namespace painted_cells_theorem_l3964_396450

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 → 
  (((2 * k + 1) * (2 * l + 1) - 74 = 373) ∨ 
   ((2 * k + 1) * (2 * l + 1) - 74 = 301)) := by
  sorry

end painted_cells_theorem_l3964_396450


namespace brennan_pepper_usage_l3964_396412

/-- The amount of pepper Brennan used for scrambled eggs -/
def pepper_used (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

/-- Theorem: Given Brennan's initial and remaining pepper amounts, prove he used 0.16 grams for scrambled eggs -/
theorem brennan_pepper_usage :
  let initial : ℝ := 0.25
  let remaining : ℝ := 0.09
  pepper_used initial remaining = 0.16 := by
  sorry

end brennan_pepper_usage_l3964_396412


namespace cost_price_percentage_l3964_396475

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) : 
  marked_price > 0 →
  cost_price > 0 →
  selling_price = marked_price * 0.9 →
  selling_price = cost_price * (1 + 20 / 700) →
  cost_price / marked_price = 0.875 := by
sorry

end cost_price_percentage_l3964_396475


namespace abs_negative_two_l3964_396418

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end abs_negative_two_l3964_396418


namespace average_allowance_proof_l3964_396413

theorem average_allowance_proof (total_students : ℕ) (total_amount : ℚ) 
  (h1 : total_students = 60)
  (h2 : total_amount = 320)
  (h3 : (2 : ℚ) / 3 * total_students + (1 : ℚ) / 3 * total_students = total_students)
  (h4 : (1 : ℚ) / 3 * total_students * 4 + (2 : ℚ) / 3 * total_students * x = total_amount) :
  x = 6 := by
  sorry

end average_allowance_proof_l3964_396413


namespace fraction_problem_l3964_396427

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end fraction_problem_l3964_396427


namespace z_in_fourth_quadrant_l3964_396402

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem z_in_fourth_quadrant :
  let z : ℂ := 3 / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end z_in_fourth_quadrant_l3964_396402


namespace two_dogs_food_consumption_l3964_396488

/-- The amount of dog food eaten by two dogs per day -/
def total_dog_food (dog1_food : ℝ) (dog2_food : ℝ) : ℝ :=
  dog1_food + dog2_food

/-- Theorem stating that two dogs eating 0.125 scoops each consume 0.25 scoops in total -/
theorem two_dogs_food_consumption :
  total_dog_food 0.125 0.125 = 0.25 := by
  sorry

end two_dogs_food_consumption_l3964_396488


namespace other_solution_of_quadratic_l3964_396425

theorem other_solution_of_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 29 = 35 * (3/4) + 12) → 
  (48 * (1/3)^2 + 29 = 35 * (1/3) + 12) := by
  sorry

end other_solution_of_quadratic_l3964_396425


namespace external_diagonal_inequality_l3964_396498

theorem external_diagonal_inequality (a b c x y z : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  x^2 = a^2 + b^2 ∧ y^2 = b^2 + c^2 ∧ z^2 = a^2 + c^2 →
  x^2 + y^2 ≥ z^2 ∧ y^2 + z^2 ≥ x^2 ∧ z^2 + x^2 ≥ y^2 := by sorry

end external_diagonal_inequality_l3964_396498


namespace sqrt_product_plus_ten_l3964_396428

theorem sqrt_product_plus_ten : Real.sqrt 18 * Real.sqrt 32 + 10 = 34 := by
  sorry

end sqrt_product_plus_ten_l3964_396428


namespace nathan_total_earnings_l3964_396485

/-- Nathan's hourly wage in dollars -/
def hourly_wage : ℝ := 6

/-- Hours worked in the second week of July -/
def hours_week2 : ℝ := 12

/-- Hours worked in the third week of July -/
def hours_week3 : ℝ := 18

/-- Earnings difference between the third and second week -/
def earnings_difference : ℝ := 36

theorem nathan_total_earnings : 
  hourly_wage * hours_week2 + hourly_wage * hours_week3 = 180 := by
  sorry

end nathan_total_earnings_l3964_396485


namespace sum_local_values_2345_l3964_396493

def local_value (digit : Nat) (place : Nat) : Nat := digit * (10 ^ place)

theorem sum_local_values_2345 :
  let thousands := local_value 2 3
  let hundreds := local_value 3 2
  let tens := local_value 4 1
  let ones := local_value 5 0
  thousands + hundreds + tens + ones = 2345 := by
sorry

end sum_local_values_2345_l3964_396493


namespace larry_wins_probability_l3964_396494

theorem larry_wins_probability (larry_prob julius_prob : ℚ) : 
  larry_prob = 3/5 →
  julius_prob = 2/5 →
  let win_prob := larry_prob / (1 - (1 - larry_prob) * (1 - julius_prob))
  win_prob = 11/15 := by
sorry

end larry_wins_probability_l3964_396494


namespace boat_round_trip_time_l3964_396464

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 15)
  (h2 : stream_speed = 3)
  (h3 : distance = 180)
  : (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 25 := by
  sorry

end boat_round_trip_time_l3964_396464


namespace sequence_sum_and_kth_term_l3964_396440

theorem sequence_sum_and_kth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (k : ℕ) 
  (h1 : ∀ n, S n = n^2 - 8*n) 
  (h2 : a k = 5) : 
  k = 7 := by sorry

end sequence_sum_and_kth_term_l3964_396440


namespace x_eighth_power_is_one_l3964_396431

theorem x_eighth_power_is_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end x_eighth_power_is_one_l3964_396431


namespace not_first_class_probability_l3964_396417

theorem not_first_class_probability (A : Set α) (P : Set α → ℝ) 
  (h1 : P A = 0.65) : P (Aᶜ) = 0.35 := by
  sorry

end not_first_class_probability_l3964_396417


namespace laptop_price_calculation_l3964_396446

/-- Calculate the total selling price of a laptop given the original price, discount rate, coupon value, and tax rate -/
def totalSellingPrice (originalPrice discountRate couponValue taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let priceAfterCoupon := discountedPrice - couponValue
  let finalPrice := priceAfterCoupon * (1 + taxRate)
  finalPrice

/-- Theorem stating that the total selling price of the laptop is 908.5 dollars -/
theorem laptop_price_calculation :
  totalSellingPrice 1200 0.30 50 0.15 = 908.5 := by
  sorry


end laptop_price_calculation_l3964_396446


namespace min_sum_positive_integers_l3964_396426

theorem min_sum_positive_integers (a b x y z : ℕ+) 
  (h : (3 : ℕ) * a.val = (7 : ℕ) * b.val ∧ 
       (7 : ℕ) * b.val = (5 : ℕ) * x.val ∧ 
       (5 : ℕ) * x.val = (4 : ℕ) * y.val ∧ 
       (4 : ℕ) * y.val = (6 : ℕ) * z.val) : 
  a.val + b.val + x.val + y.val + z.val ≥ 459 := by
  sorry

end min_sum_positive_integers_l3964_396426


namespace four_three_three_cuboid_two_face_count_l3964_396442

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  2 * (c.length - 2) + 2 * (c.width - 2) + 2 * (c.height - 2)

/-- Theorem: A 4x3x3 cuboid has 16 cubes with exactly two painted faces -/
theorem four_three_three_cuboid_two_face_count :
  count_two_face_cubes ⟨4, 3, 3⟩ = 16 := by
  sorry

end four_three_three_cuboid_two_face_count_l3964_396442


namespace set_D_is_empty_l3964_396486

def set_D : Set ℝ := {x : ℝ | x^2 - x + 1 = 0}

theorem set_D_is_empty : set_D = ∅ := by
  sorry

end set_D_is_empty_l3964_396486


namespace largest_sum_is_923_l3964_396499

def digits : List Nat := [3, 5, 7, 8, 0]

def is_valid_partition (a b : List Nat) : Prop :=
  a.length = 3 ∧ b.length = 2 ∧ (a ++ b).toFinset = digits.toFinset

def to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem largest_sum_is_923 :
  ∀ a b : List Nat,
    is_valid_partition a b →
    to_number a + to_number b ≤ 923 :=
by sorry

end largest_sum_is_923_l3964_396499


namespace tomorrow_is_saturday_l3964_396457

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (addDays d k)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : addDays Day.Wednesday 5 = Day.Monday) : 
  nextDay Day.Friday = Day.Saturday :=
by sorry

end tomorrow_is_saturday_l3964_396457


namespace sum_of_fourth_and_fifth_terms_l3964_396447

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_of_fourth_and_fifth_terms :
  ∀ (a₀ r : ℝ),
    geometric_sequence a₀ r 0 = 4096 →
    geometric_sequence a₀ r 1 = 1024 →
    geometric_sequence a₀ r 2 = 256 →
    geometric_sequence a₀ r 5 = 4 →
    geometric_sequence a₀ r 6 = 1 →
    geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80 := by
  sorry

end sum_of_fourth_and_fifth_terms_l3964_396447


namespace monic_quartic_with_specific_roots_l3964_396460

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 25*x^2 + 2*x - 12

-- State the theorem
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 10*x^3 + 25*x^2 + 2*x - 12) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2 - √7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry


end monic_quartic_with_specific_roots_l3964_396460


namespace decreasing_number_a312_max_decreasing_number_divisible_by_9_l3964_396451

/-- A four-digit natural number with all digits different and not equal to 0 -/
structure DecreasingNumber :=
  (a b c d : ℕ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (decreasing_property : 10 * a + b - (10 * b + c) = 10 * c + d)

theorem decreasing_number_a312 :
  ∃ (n : DecreasingNumber), n.a = 4 ∧ n.b = 3 ∧ n.c = 1 ∧ n.d = 2 :=
sorry

theorem max_decreasing_number_divisible_by_9 :
  ∃ (n : DecreasingNumber),
    (100 * n.a + 10 * n.b + n.c + 100 * n.b + 10 * n.c + n.d) % 9 = 0 ∧
    ∀ (m : DecreasingNumber),
      (100 * m.a + 10 * m.b + m.c + 100 * m.b + 10 * m.c + m.d) % 9 = 0 →
      1000 * n.a + 100 * n.b + 10 * n.c + n.d ≥ 1000 * m.a + 100 * m.b + 10 * m.c + m.d ∧
    n.a = 8 ∧ n.b = 1 ∧ n.c = 6 ∧ n.d = 5 :=
sorry

end decreasing_number_a312_max_decreasing_number_divisible_by_9_l3964_396451


namespace quadratic_function_property_l3964_396422

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property
  (a b c : ℝ) (h : a ≠ 0)
  (x₁ x₂ : ℝ) (hx : x₁ ≠ x₂)
  (hf : QuadraticFunction a b c h x₁ = QuadraticFunction a b c h x₂) :
  QuadraticFunction a b c h (x₁ + x₂) = c := by
  sorry

end quadratic_function_property_l3964_396422


namespace function_properties_l3964_396432

-- Define the function f(x) = -x^2 + mx - m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  -- 1. If the maximum value of f(x) is 0, then m = 0 or m = 4
  (∃ (max : ℝ), (∀ (x : ℝ), f m x ≤ max) ∧ (max = 0)) →
  (m = 0 ∨ m = 4) ∧

  -- 2. If f(x) is monotonically decreasing on [-1, 0], then m ≤ -2
  (∀ (x y : ℝ), -1 ≤ x ∧ x < y ∧ y ≤ 0 → f m x > f m y) →
  (m ≤ -2) ∧

  -- 3. The range of f(x) on [2, 3] is exactly [2, 3] if and only if m = 6
  (∀ (y : ℝ), 2 ≤ y ∧ y ≤ 3 ↔ ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 3 ∧ f m x = y) ↔
  (m = 6) :=
by sorry

end function_properties_l3964_396432


namespace total_pictures_correct_l3964_396441

/-- The number of pictures Nancy uploaded to Facebook -/
def total_pictures : ℕ := 51

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 11

/-- The number of additional albums -/
def additional_albums : ℕ := 8

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 5

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end total_pictures_correct_l3964_396441


namespace tangent_line_at_2_sum_formula_min_value_nSn_l3964_396487

/-- The original function -/
def g (x : ℝ) : ℝ := x^2 - 2*x - 11

/-- The tangent line to g(x) at x = 2 -/
def f (x : ℝ) : ℝ := 2*x - 15

/-- The sequence a_n -/
def a (n : ℕ) : ℝ := f n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := n^2 - 14*n

theorem tangent_line_at_2 : 
  ∀ x, f x = (2 : ℝ) * (x - 2) + g 2 :=
sorry

theorem sum_formula : 
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2 :=
sorry

theorem min_value_nSn : 
  ∃ n : ℕ, ∀ m : ℕ, m ≥ 1 → (n : ℝ) * S n ≤ (m : ℝ) * S m ∧ 
  (n : ℝ) * S n = -405 :=
sorry

end tangent_line_at_2_sum_formula_min_value_nSn_l3964_396487


namespace nelly_paid_correct_amount_l3964_396434

/-- Nelly's payment for a painting at an auction -/
def nellys_payment (joe_bid sarah_bid : ℕ) : ℕ :=
  max
    (3 * joe_bid + 2000)
    (4 * sarah_bid + 1500)

/-- Theorem stating the correct amount Nelly paid for the painting -/
theorem nelly_paid_correct_amount :
  nellys_payment 160000 50000 = 482000 := by
  sorry

end nelly_paid_correct_amount_l3964_396434


namespace bridge_length_bridge_length_proof_l3964_396437

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_proof :
  bridge_length 80 45 30 = 295 := by
  sorry

end bridge_length_bridge_length_proof_l3964_396437


namespace concert_ticket_sales_l3964_396495

theorem concert_ticket_sales
  (student_price : ℕ)
  (non_student_price : ℕ)
  (total_revenue : ℕ)
  (student_tickets : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_revenue = 20960)
  (h4 : student_tickets = 520) :
  ∃ (non_student_tickets : ℕ),
    student_tickets * student_price + non_student_tickets * non_student_price = total_revenue ∧
    student_tickets + non_student_tickets = 2000 :=
by
  sorry

#check concert_ticket_sales

end concert_ticket_sales_l3964_396495


namespace cubic_function_properties_l3964_396470

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line parallel to 3x + y + 2 = 0 at x = 1
  (a = -1 ∧ b = 0) ∧  -- Values of a and b
  (∃ x₁ x₂ : ℝ, f a b c x₁ - f a b c x₂ = 4)  -- Difference between max and min is 4
  := by sorry

end cubic_function_properties_l3964_396470


namespace solution_product_l3964_396410

theorem solution_product (p q : ℝ) : 
  p ≠ q ∧ 
  (p - 7) * (3 * p + 11) = p^2 - 20 * p + 63 ∧ 
  (q - 7) * (3 * q + 11) = q^2 - 20 * q + 63 →
  (p + 2) * (q + 2) = -72 := by
sorry

end solution_product_l3964_396410


namespace cards_per_layer_calculation_l3964_396466

def number_of_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def number_of_layers : ℕ := 32

def total_cards : ℕ := number_of_decks * cards_per_deck

theorem cards_per_layer_calculation :
  total_cards / number_of_layers = 26 := by sorry

end cards_per_layer_calculation_l3964_396466


namespace prob_ace_king_same_suit_standard_deck_l3964_396456

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (aces_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Probability of drawing an Ace then a King of the same suit -/
def prob_ace_then_king_same_suit (d : Deck) : ℚ :=
  (d.aces_per_suit : ℚ) / d.total_cards * (d.kings_per_suit : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace then a King of the same suit in a standard deck -/
theorem prob_ace_king_same_suit_standard_deck :
  let standard_deck : Deck :=
    { total_cards := 52
    , suits := 4
    , cards_per_suit := 13
    , aces_per_suit := 1
    , kings_per_suit := 1
    }
  prob_ace_then_king_same_suit standard_deck = 1 / 663 := by
  sorry


end prob_ace_king_same_suit_standard_deck_l3964_396456


namespace descendant_divisibility_l3964_396471

theorem descendant_divisibility (N : ℕ) (h : N ≥ 10000 ∧ N < 100000) :
  N % 271 = 0 → (N * 10 + N / 10000 - (N / 10000) * 100000) % 271 = 0 := by
  sorry

end descendant_divisibility_l3964_396471


namespace lisa_candies_on_specific_days_l3964_396454

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_specific_days (total_candies : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (specific_days : ℕ) : ℕ :=
  let candies_on_other_days := (days_per_week - specific_days) * weeks
  let remaining_candies := total_candies - candies_on_other_days
  remaining_candies / (specific_days * weeks)

/-- Theorem stating that Lisa eats 2 candies on Mondays and Wednesdays -/
theorem lisa_candies_on_specific_days : 
  candies_on_specific_days 36 4 7 2 = 2 := by
  sorry

end lisa_candies_on_specific_days_l3964_396454


namespace unique_solution_modular_system_l3964_396444

theorem unique_solution_modular_system :
  ∃! x : ℕ, x < 12 ∧ (5 * x + 3) % 15 = 7 % 15 ∧ x % 4 = 2 % 4 :=
by sorry

end unique_solution_modular_system_l3964_396444


namespace odd_sum_product_equivalence_l3964_396407

theorem odd_sum_product_equivalence (p q : ℕ) 
  (hp : p < 16 ∧ p % 2 = 1) 
  (hq : q < 16 ∧ q % 2 = 1) : 
  p * q + p + q = (p + 1) * (q + 1) - 1 := by
  sorry

end odd_sum_product_equivalence_l3964_396407


namespace expand_expression_l3964_396436

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x := by
  sorry

end expand_expression_l3964_396436


namespace product_xyz_l3964_396467

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2 * y = z) 
  (h2 : x^2 - 4 * y^2 + z^2 = 310) : 
  x * y * z = 11935 ∨ x * y * z = 2015 := by
  sorry

end product_xyz_l3964_396467


namespace indeterminate_value_l3964_396416

theorem indeterminate_value (a b c d : ℝ) : 
  (b - d)^2 = 4 → 
  ¬∃!x, x = a + b - c - d :=
by sorry

end indeterminate_value_l3964_396416


namespace cafeteria_pies_l3964_396476

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 75 →
  handed_out = 19 →
  apples_per_pie = 8 →
  (initial_apples - handed_out) / apples_per_pie = 7 := by
  sorry

end cafeteria_pies_l3964_396476


namespace rectangle_perimeter_l3964_396484

theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 576 →
  2 * (width + length) = 72 * Real.sqrt 2 :=
by sorry

end rectangle_perimeter_l3964_396484


namespace simplify_fraction_l3964_396403

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end simplify_fraction_l3964_396403


namespace three_sets_sum_18_with_6_l3964_396477

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem three_sets_sum_18_with_6 :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    s ⊆ S ∧ 
    6 ∈ s ∧ 
    s.sum id = 18
  ) (S.powerset)).card = 3 := by
  sorry

end three_sets_sum_18_with_6_l3964_396477


namespace installation_cost_calculation_l3964_396490

/-- Calculates the installation cost given the purchase details of a refrigerator. -/
theorem installation_cost_calculation
  (purchase_price_after_discount : ℝ)
  (discount_rate : ℝ)
  (transport_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price_after_discount = 12500)
  (h2 : discount_rate = 0.20)
  (h3 : transport_cost = 125)
  (h4 : selling_price = 18400)
  (h5 : selling_price = 1.15 * (purchase_price_after_discount + transport_cost + installation_cost)) :
  installation_cost = 3375 :=
by sorry


end installation_cost_calculation_l3964_396490


namespace uncovered_volume_is_229_l3964_396430

def shoebox_volume : ℝ := 4 * 6 * 12

def object1_volume : ℝ := 5 * 3 * 1
def object2_volume : ℝ := 2 * 2 * 3
def object3_volume : ℝ := 4 * 2 * 4

def total_object_volume : ℝ := object1_volume + object2_volume + object3_volume

theorem uncovered_volume_is_229 : 
  shoebox_volume - total_object_volume = 229 := by sorry

end uncovered_volume_is_229_l3964_396430


namespace no_horizontal_asymptote_l3964_396482

noncomputable def f (x : ℝ) : ℝ :=
  (18 * x^5 + 12 * x^4 + 4 * x^3 + 9 * x^2 + 5 * x + 3) /
  (3 * x^4 + 2 * x^3 + 8 * x^2 + 3 * x + 1)

theorem no_horizontal_asymptote :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ x > N, |f x - L| < ε :=
by
  sorry

end no_horizontal_asymptote_l3964_396482


namespace lemonade_sales_l3964_396420

theorem lemonade_sales (last_week : ℝ) (this_week : ℝ) (total : ℝ) : 
  this_week = 1.3 * last_week →
  total = last_week + this_week →
  total = 46 →
  last_week = 20 := by
sorry

end lemonade_sales_l3964_396420


namespace product_of_fractions_l3964_396411

theorem product_of_fractions : (2 : ℚ) / 5 * (3 : ℚ) / 4 = (3 : ℚ) / 10 := by
  sorry

end product_of_fractions_l3964_396411


namespace bicycle_weight_proof_l3964_396424

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 20

/-- The weight of one scooter in pounds -/
def scooter_weight : ℝ := 40

theorem bicycle_weight_proof :
  (10 * bicycle_weight = 5 * scooter_weight) ∧
  (5 * scooter_weight = 200) →
  bicycle_weight = 20 :=
by
  sorry


end bicycle_weight_proof_l3964_396424


namespace root_implies_a_in_interval_l3964_396445

/-- Given that for all real m, the function f(x) = m(x^2 - 1) + x - a always has a root,
    prove that a is in the interval [-1, 1] -/
theorem root_implies_a_in_interval :
  (∀ m : ℝ, ∃ x : ℝ, m * (x^2 - 1) + x - a = 0) →
  a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end root_implies_a_in_interval_l3964_396445


namespace simplify_expression_l3964_396453

theorem simplify_expression (p : ℝ) :
  ((6*p + 2) - 3*p*3)*4 + (5 - 2/4)*(8*p - 12) = 24*p - 46 := by
  sorry

end simplify_expression_l3964_396453


namespace instantaneous_acceleration_at_3s_l3964_396469

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Define the acceleration function as the derivative of velocity
def acceleration (t : ℝ) : ℝ := 12 * t

-- Theorem statement
theorem instantaneous_acceleration_at_3s :
  acceleration 3 = 36 := by
  sorry


end instantaneous_acceleration_at_3s_l3964_396469


namespace min_value_x_plus_2y_l3964_396481

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
sorry

end min_value_x_plus_2y_l3964_396481


namespace eventually_constant_l3964_396479

/-- The set of positive integers -/
def PositiveInts : Set ℕ := {n : ℕ | n > 0}

/-- The winning set for (n,S)-nim game -/
def winning_set (S : Set ℕ) : Set ℕ :=
  {n : ℕ | ∃ (strategy : ℕ → ℕ), ∀ (m : ℕ), m < n → strategy m ∈ S ∧ strategy m ≤ n}

/-- The function f that maps a set S to its winning set -/
def f (S : Set ℕ) : Set ℕ := winning_set S

/-- Iterate f k times -/
def iterate_f (S : Set ℕ) : ℕ → Set ℕ
  | 0 => S
  | k + 1 => f (iterate_f S k)

/-- The main theorem: the sequence of iterations of f eventually becomes constant -/
theorem eventually_constant (T : Set ℕ) : ∃ (k : ℕ), iterate_f T k = iterate_f T (k + 1) := by
  sorry

end eventually_constant_l3964_396479


namespace power_division_multiplication_l3964_396408

theorem power_division_multiplication : (8^3 / 8^2) * 3^3 = 216 := by
  sorry

end power_division_multiplication_l3964_396408


namespace max_k_value_l3964_396497

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  ∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k :=
sorry

end max_k_value_l3964_396497


namespace negation_of_existence_negation_of_quadratic_inequality_l3964_396415

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 > 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3964_396415


namespace peanut_butter_calories_value_l3964_396473

/-- The number of calories in a serving of peanut butter -/
def peanut_butter_calories : ℕ := sorry

/-- The number of calories in a piece of bread -/
def bread_calories : ℕ := 100

/-- The total number of calories for breakfast -/
def total_calories : ℕ := 500

/-- The number of servings of peanut butter -/
def peanut_butter_servings : ℕ := 2

/-- The number of pieces of bread -/
def bread_pieces : ℕ := 1

theorem peanut_butter_calories_value : 
  bread_calories * bread_pieces + peanut_butter_calories * peanut_butter_servings = total_calories ∧ 
  peanut_butter_calories = 200 := by sorry

end peanut_butter_calories_value_l3964_396473


namespace volunteer_schedule_l3964_396448

theorem volunteer_schedule (ella fiona george harry : ℕ) 
  (h_ella : ella = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm ella fiona) george) harry = 360 := by
  sorry

end volunteer_schedule_l3964_396448


namespace shape_is_cone_l3964_396419

/-- The shape described by ρ = c sin φ in spherical coordinates is a cone -/
theorem shape_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)),
    ∀ (ρ θ φ : ℝ),
      (ρ, θ, φ) ∈ cone ↔ ρ = c * Real.sin φ ∧ ρ ≥ 0 ∧ θ ∈ Set.Icc 0 (2 * Real.pi) ∧ φ ∈ Set.Icc 0 Real.pi :=
by sorry

end shape_is_cone_l3964_396419


namespace square_area_equal_perimeter_triangle_l3964_396461

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (h_triangle : a = 7.2 ∧ b = 9.5 ∧ c = 11.3) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 49 := by
sorry

end square_area_equal_perimeter_triangle_l3964_396461


namespace parabola_theorem_l3964_396414

/-- Represents a parabola in the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The parabola passes through the point (2, 0) -/
def passes_through_A (p : Parabola) : Prop :=
  4 + 2 * p.b + p.c = 0

/-- The parabola passes through the point (0, 6) -/
def passes_through_B (p : Parabola) : Prop :=
  p.c = 6

/-- The parabola equation is y = x^2 - 5x + 6 -/
def is_correct_equation (p : Parabola) : Prop :=
  p.b = -5 ∧ p.c = 6

/-- The y-coordinate of the point (4, 0) on the parabola -/
def y_at_x_4 (p : Parabola) : ℝ :=
  16 - 5 * 4 + p.c

/-- The downward shift required for the parabola to pass through (4, 0) -/
def downward_shift (p : Parabola) : ℝ :=
  y_at_x_4 p

theorem parabola_theorem (p : Parabola) 
  (h1 : passes_through_A p) (h2 : passes_through_B p) : 
  is_correct_equation p ∧ downward_shift p = 2 := by
  sorry

end parabola_theorem_l3964_396414


namespace possible_values_of_a_l3964_396421

def P : Set ℝ := {x | x^2 ≠ 1}

def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a : 
  ∀ a : ℝ, M a ⊆ P ↔ a ∈ ({1, -1, 0} : Set ℝ) := by sorry

end possible_values_of_a_l3964_396421


namespace unique_valid_number_l3964_396443

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 100 % 10 > 6) ∧ (n / 10 % 10 > 6) ∧ (n % 10 > 6) ∧
  n % 12 = 0

theorem unique_valid_number : ∃! n, is_valid_number n :=
sorry

end unique_valid_number_l3964_396443


namespace vector_magnitude_equation_l3964_396468

theorem vector_magnitude_equation (k : ℝ) : 
  ‖k • (⟨3, -4⟩ : ℝ × ℝ) + ⟨5, -6⟩‖ = 5 * Real.sqrt 5 ↔ k = 17/25 ∨ k = -19/5 := by
  sorry

end vector_magnitude_equation_l3964_396468


namespace total_distance_in_land_miles_l3964_396483

/-- Represents the speed of the sailboat in knots -/
structure SailboatSpeed where
  oneSail : ℝ
  twoSails : ℝ

/-- Represents the travel time in hours -/
structure TravelTime where
  oneSail : ℝ
  twoSails : ℝ

/-- Conversion factors -/
def knotToNauticalMile : ℝ := 1
def nauticalMileToLandMile : ℝ := 1.15

theorem total_distance_in_land_miles 
  (speed : SailboatSpeed) 
  (time : TravelTime) 
  (h1 : speed.oneSail = 25)
  (h2 : speed.twoSails = 50)
  (h3 : time.oneSail = 4)
  (h4 : time.twoSails = 4) :
  (speed.oneSail * time.oneSail + speed.twoSails * time.twoSails) * 
  knotToNauticalMile * nauticalMileToLandMile = 345 := by
  sorry

#check total_distance_in_land_miles

end total_distance_in_land_miles_l3964_396483


namespace tanya_plums_l3964_396465

/-- The number of plums Tanya bought at the grocery store -/
def plums : ℕ := 6

/-- The total number of pears, apples, and pineapples Tanya bought -/
def other_fruits : ℕ := 12

/-- The number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_plums :
  plums = remaining_fruits * 2 - other_fruits :=
by sorry

end tanya_plums_l3964_396465


namespace samson_schedule_solution_l3964_396491

/-- Utility function -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := 2 * math * frisbee

/-- Wednesday's utility -/
def wednesday_utility (s : ℝ) : ℝ := utility (10 - 2*s) s

/-- Thursday's utility -/
def thursday_utility (s : ℝ) : ℝ := utility (2*s + 4) (3 - s)

/-- The theorem stating that s = 2 is the unique solution -/
theorem samson_schedule_solution :
  ∃! s : ℝ, wednesday_utility s = thursday_utility s ∧ s = 2 := by
  sorry

end samson_schedule_solution_l3964_396491


namespace inlet_pipe_rate_l3964_396480

/-- Given a tank with specified capacity and emptying times, calculate the inlet pipe rate -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (outlet_time : ℝ) (combined_time : ℝ) :
  tank_capacity = 3200 →
  outlet_time = 5 →
  combined_time = 8 →
  (tank_capacity / combined_time - tank_capacity / outlet_time) * (1 / 60) = 4 := by
  sorry

end inlet_pipe_rate_l3964_396480


namespace vector_perpendicular_problem_l3964_396459

theorem vector_perpendicular_problem (x : ℝ) : 
  let a : ℝ × ℝ := (-2, x)
  let b : ℝ × ℝ := (1, Real.sqrt 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = 2 * Real.sqrt 3 := by
  sorry

end vector_perpendicular_problem_l3964_396459


namespace cos_135_degrees_l3964_396409

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l3964_396409


namespace symmetric_line_proof_l3964_396423

/-- Given two lines in a 2D plane, this function returns the equation of the line symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  λ x y ↦ x - y - 2 = 0

/-- The line x - 2y + 2 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  λ x y ↦ x - 2*y + 2 = 0

/-- The line x - 7y + 22 = 0 -/
def resultLine : ℝ → ℝ → Prop :=
  λ x y ↦ x - 7*y + 22 = 0

theorem symmetric_line_proof : 
  symmetricLine line1 line2 = resultLine := by
  sorry

end symmetric_line_proof_l3964_396423


namespace total_players_specific_l3964_396478

/-- The number of players in a sports event with overlapping groups --/
def totalPlayers (kabadi khoKho soccer kabadi_khoKho kabadi_soccer khoKho_soccer all_three : ℕ) : ℕ :=
  kabadi + khoKho + soccer - kabadi_khoKho - kabadi_soccer - khoKho_soccer + all_three

/-- Theorem stating the total number of players given the specific conditions --/
theorem total_players_specific : totalPlayers 50 80 30 15 10 25 8 = 118 := by
  sorry

end total_players_specific_l3964_396478


namespace problem_statement_l3964_396462

theorem problem_statement (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 7.2 := by
  sorry

end problem_statement_l3964_396462


namespace arithmetic_geometric_mean_inequality_l3964_396472

theorem arithmetic_geometric_mean_inequality 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2 * Real.sqrt (a * b) := by
  sorry

end arithmetic_geometric_mean_inequality_l3964_396472


namespace smallest_valid_n_l3964_396463

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧ ∀ p, Nat.Prime p → p ∣ n → n > 1200 * p

theorem smallest_valid_n :
  is_valid 3888 ∧ ∀ m, m < 3888 → ¬is_valid m :=
sorry

end smallest_valid_n_l3964_396463


namespace truck_travel_distance_truck_specific_distance_l3964_396438

/-- Represents the distance a truck can travel given an amount of gas -/
def distance_traveled (miles_per_gallon : ℝ) (gallons : ℝ) : ℝ :=
  miles_per_gallon * gallons

theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_gas : ℝ) 
  (new_gas : ℝ) : 
  initial_distance > 0 → 
  initial_gas > 0 → 
  new_gas > 0 → 
  distance_traveled (initial_distance / initial_gas) new_gas = 
    (initial_distance / initial_gas) * new_gas := by
  sorry

/-- Proves that a truck traveling 240 miles on 10 gallons of gas can travel 360 miles on 15 gallons of gas -/
theorem truck_specific_distance : 
  distance_traveled (240 / 10) 15 = 360 := by
  sorry

end truck_travel_distance_truck_specific_distance_l3964_396438


namespace original_sheet_area_l3964_396433

/-- Represents the dimensions and properties of a cardboard box created from a rectangular sheet. -/
structure CardboardBox where
  base_length : ℝ
  base_width : ℝ
  volume : ℝ

/-- Theorem stating that given the specified conditions, the original sheet area is 110 cm². -/
theorem original_sheet_area
  (box : CardboardBox)
  (base_length_eq : box.base_length = 5)
  (base_width_eq : box.base_width = 4)
  (volume_eq : box.volume = 60)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check original_sheet_area

end original_sheet_area_l3964_396433


namespace symmetric_point_wrt_origin_l3964_396406

/-- 
Given a point P with coordinates (2, -5), 
prove that its symmetric point P' with respect to the origin has coordinates (-2, 5).
-/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-2, 5) := by sorry

end symmetric_point_wrt_origin_l3964_396406
