import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4087_408772

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (a : ℕ) + (b : ℕ) ≥ (x : ℕ) + (y : ℕ)) → 
  (x : ℕ) + (y : ℕ) = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l4087_408772


namespace NUMINAMATH_CALUDE_students_in_all_classes_l4087_408771

/-- Represents the number of students registered for a combination of classes -/
structure ClassRegistration where
  history : ℕ
  math : ℕ
  english : ℕ
  historyMath : ℕ
  historyEnglish : ℕ
  mathEnglish : ℕ
  allThree : ℕ

/-- The theorem stating the number of students registered for all three classes -/
theorem students_in_all_classes 
  (total : ℕ) 
  (classes : ClassRegistration) 
  (h1 : total = 86)
  (h2 : classes.history = 12)
  (h3 : classes.math = 17)
  (h4 : classes.english = 36)
  (h5 : classes.historyMath + classes.historyEnglish + classes.mathEnglish = 3)
  (h6 : total = classes.history + classes.math + classes.english - 
        (classes.historyMath + classes.historyEnglish + classes.mathEnglish) + 
        classes.allThree) :
  classes.allThree = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_in_all_classes_l4087_408771


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l4087_408729

/-- An eight-digit number in the form 8524m637 -/
def eight_digit_number (m : ℕ) : ℕ := 85240000 + m * 1000 + 637

/-- Sum of digits in odd positions -/
def sum_odd_digits (m : ℕ) : ℕ := 8 + 2 + m + 3

/-- Sum of digits in even positions -/
def sum_even_digits : ℕ := 5 + 4 + 6 + 7

/-- A number is divisible by 11 if the difference between the sum of digits in odd and even positions is a multiple of 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (sum_odd_digits n - sum_even_digits : ℤ) = 11 * k

theorem eight_digit_divisible_by_11 :
  ∃ m : ℕ, m < 10 ∧ divisible_by_11 (eight_digit_number m) ↔ m = 9 :=
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l4087_408729


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4087_408785

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-2) 1 ∧ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4087_408785


namespace NUMINAMATH_CALUDE_num_distinct_configurations_l4087_408796

/-- The group of cube rotations -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of cube rotations -/
def numRotations : ℕ := 4

/-- The number of configurations fixed by the identity rotation -/
def fixedByIdentity : ℕ := 56

/-- The number of configurations fixed by each 180-degree rotation -/
def fixedBy180Rotation : ℕ := 6

/-- The number of 180-degree rotations -/
def num180Rotations : ℕ := 3

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := fixedByIdentity + num180Rotations * fixedBy180Rotation

/-- The theorem stating the number of distinct configurations -/
theorem num_distinct_configurations : 
  (totalFixedPoints : ℚ) / numRotations = 19 / 2 := by sorry

end NUMINAMATH_CALUDE_num_distinct_configurations_l4087_408796


namespace NUMINAMATH_CALUDE_odd_sequence_sum_l4087_408747

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n / 2 * (a₁ + aₙ)

theorem odd_sequence_sum :
  ∃ (n : ℕ), 
    let a₁ := 1
    let aₙ := 79
    let sum := arithmetic_sum a₁ aₙ n
    n > 0 ∧ aₙ = a₁ + 2 * (n - 1) ∧ 3 * sum = 4800 := by
  sorry

end NUMINAMATH_CALUDE_odd_sequence_sum_l4087_408747


namespace NUMINAMATH_CALUDE_square_equation_solve_l4087_408761

theorem square_equation_solve (x y : ℝ) (h1 : x^2 = y + 4) (h2 : x = 7) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solve_l4087_408761


namespace NUMINAMATH_CALUDE_add_like_terms_l4087_408714

theorem add_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l4087_408714


namespace NUMINAMATH_CALUDE_even_number_representation_l4087_408776

theorem even_number_representation (x y : ℕ) : 
  ∃! n : ℕ, 2 * n = (x + y)^2 + 3 * x + y := by sorry

end NUMINAMATH_CALUDE_even_number_representation_l4087_408776


namespace NUMINAMATH_CALUDE_first_divisor_problem_l4087_408728

theorem first_divisor_problem (y : ℝ) (h : (320 / y) / 3 = 53.33) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l4087_408728


namespace NUMINAMATH_CALUDE_pizza_toppings_l4087_408754

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 10 →
  cheese_slices = 5 →
  mushroom_slices = 7 →
  cheese_slices + mushroom_slices - total_slices = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l4087_408754


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l4087_408768

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 823 = 0 ∧
  (n + 1) % 618 = 0 ∧
  (n + 1) % 3648 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 3917 = 0 ∧
  (n + 1) % 4203 = 0

theorem smallest_number_divisible_by_all :
  ∃ n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l4087_408768


namespace NUMINAMATH_CALUDE_decorative_gravel_cost_l4087_408786

/-- The cost of decorative gravel in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of gravel -/
def cubic_yards : ℝ := 8

/-- The total cost of the decorative gravel -/
def total_cost : ℝ := cubic_yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem decorative_gravel_cost : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_decorative_gravel_cost_l4087_408786


namespace NUMINAMATH_CALUDE_log_intersection_and_exponential_inequality_l4087_408715

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the inverse function of f (exponential function)
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem log_intersection_and_exponential_inequality :
  (∃! x : ℝ, f x = x - 1) ∧
  (∀ m n : ℝ, m < n → (g n - g m) / (n - m) > g ((m + n) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_intersection_and_exponential_inequality_l4087_408715


namespace NUMINAMATH_CALUDE_sugar_calculation_l4087_408784

theorem sugar_calculation (standard_sugar : ℚ) (reduced_sugar : ℚ) : 
  standard_sugar = 10/3 → 
  reduced_sugar = (1/3) * standard_sugar →
  reduced_sugar = 10/9 :=
by sorry

end NUMINAMATH_CALUDE_sugar_calculation_l4087_408784


namespace NUMINAMATH_CALUDE_mark_bought_three_weeks_of_food_l4087_408773

/-- Calculates the number of weeks of dog food purchased given the total cost,
    puppy cost, daily food consumption, bag size, and bag cost. -/
def weeks_of_food (total_cost puppy_cost daily_food_cups bag_size_cups bag_cost : ℚ) : ℚ :=
  let food_cost := total_cost - puppy_cost
  let bags_bought := food_cost / bag_cost
  let total_cups := bags_bought * bag_size_cups
  let days_of_food := total_cups / daily_food_cups
  days_of_food / 7

/-- Theorem stating that under the given conditions, Mark bought food for 3 weeks. -/
theorem mark_bought_three_weeks_of_food :
  weeks_of_food 14 10 (1/3) (7/2) 2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_mark_bought_three_weeks_of_food_l4087_408773


namespace NUMINAMATH_CALUDE_trig_identity_l4087_408795

theorem trig_identity : 
  (Real.cos (20 * π / 180)) / (Real.cos (35 * π / 180) * Real.sqrt (1 - Real.sin (20 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4087_408795


namespace NUMINAMATH_CALUDE_school_survey_most_suitable_for_census_l4087_408731

/-- Represents a survey type --/
inductive SurveyType
  | CityResidents
  | CarBatch
  | LightTubeBatch
  | SchoolStudents

/-- Determines if a survey type is suitable for a census --/
def isSuitableForCensus (s : SurveyType) : Prop :=
  match s with
  | .SchoolStudents => True
  | _ => False

/-- Theorem stating that the school students survey is the most suitable for a census --/
theorem school_survey_most_suitable_for_census :
  ∀ s : SurveyType, isSuitableForCensus s ↔ s = SurveyType.SchoolStudents :=
by sorry

end NUMINAMATH_CALUDE_school_survey_most_suitable_for_census_l4087_408731


namespace NUMINAMATH_CALUDE_train_length_l4087_408712

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4087_408712


namespace NUMINAMATH_CALUDE_barn_hoot_difference_l4087_408755

/-- The number of hoots one barnyard owl makes per minute -/
def hoots_per_owl : ℕ := 5

/-- The number of hoots heard per minute from the barn -/
def hoots_heard : ℕ := 20

/-- The number of owls we're comparing to -/
def num_owls : ℕ := 3

/-- The difference between the hoots heard and the hoots from a specific number of owls -/
def hoot_difference (heard : ℕ) (owls : ℕ) : ℤ :=
  heard - (owls * hoots_per_owl)

theorem barn_hoot_difference :
  hoot_difference hoots_heard num_owls = 5 := by
  sorry

end NUMINAMATH_CALUDE_barn_hoot_difference_l4087_408755


namespace NUMINAMATH_CALUDE_watch_cost_price_l4087_408779

def watch_problem (cost_price : ℝ) : Prop :=
  let loss_percentage : ℝ := 10
  let gain_percentage : ℝ := 4
  let additional_amount : ℝ := 210
  let selling_price_1 : ℝ := cost_price * (1 - loss_percentage / 100)
  let selling_price_2 : ℝ := cost_price * (1 + gain_percentage / 100)
  selling_price_2 = selling_price_1 + additional_amount

theorem watch_cost_price : 
  ∃ (cost_price : ℝ), watch_problem cost_price ∧ cost_price = 1500 :=
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l4087_408779


namespace NUMINAMATH_CALUDE_student_multiplication_error_l4087_408746

/-- Represents a repeating decimal of the form 1.abababab... -/
def repeating_decimal (a b : ℕ) : ℚ :=
  1 + (10 * a + b : ℚ) / 99

/-- Represents the decimal 1.ab -/
def non_repeating_decimal (a b : ℕ) : ℚ :=
  1 + (a : ℚ) / 10 + (b : ℚ) / 100

theorem student_multiplication_error (a b : ℕ) :
  a < 10 → b < 10 →
  66 * (repeating_decimal a b - non_repeating_decimal a b) = (1 : ℚ) / 2 →
  a * 10 + b = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_error_l4087_408746


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l4087_408724

/-- The cost of attractions at an amusement park. -/
structure AttractionCosts where
  roller_coaster : ℕ
  log_ride : ℕ
  ferris_wheel : ℕ

/-- The number of tickets Antonieta has and needs. -/
structure AntonietaTickets where
  current : ℕ
  needed : ℕ

/-- Theorem stating the cost of the Ferris wheel given the other costs and ticket information. -/
theorem ferris_wheel_cost (costs : AttractionCosts) (antonieta : AntonietaTickets) : 
  costs.roller_coaster = 5 →
  costs.log_ride = 7 →
  antonieta.current = 2 →
  antonieta.needed = 16 →
  costs.ferris_wheel = 6 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l4087_408724


namespace NUMINAMATH_CALUDE_position_of_2022_l4087_408777

-- Define the sequence
def sequence_element (n : ℕ) : ℕ :=
  if n % 3 = 0 then 4 * ((n - 1) / 3) + 3
  else if n % 3 = 1 then 4 * (n / 3) + 1
  else 4 * (n / 3) + 2

-- Define the group number for a given element
def group_number (x : ℕ) : ℕ :=
  (x - 1) / 3 + 1

-- Define the position within a group for a given element
def position_in_group (x : ℕ) : ℕ :=
  (x - 1) % 3 + 1

-- Theorem statement
theorem position_of_2022 :
  group_number 2022 = 506 ∧ position_in_group 2022 = 2 :=
sorry

end NUMINAMATH_CALUDE_position_of_2022_l4087_408777


namespace NUMINAMATH_CALUDE_total_animal_eyes_l4087_408733

theorem total_animal_eyes (num_frogs num_crocodiles : ℕ) 
  (eyes_per_frog eyes_per_crocodile : ℕ) : 
  num_frogs = 20 → 
  num_crocodiles = 10 → 
  eyes_per_frog = 2 → 
  eyes_per_crocodile = 2 → 
  num_frogs * eyes_per_frog + num_crocodiles * eyes_per_crocodile = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_animal_eyes_l4087_408733


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4087_408787

def is_valid_number (n : ℕ) : Prop :=
  (n % 10 = 6) ∧ 
  (∃ m : ℕ, m > 0 ∧ 6 * 10^m + n / 10 = 4 * n)

theorem smallest_valid_number : 
  (is_valid_number 1538466) ∧ 
  (∀ k < 1538466, ¬(is_valid_number k)) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4087_408787


namespace NUMINAMATH_CALUDE_min_triangles_cover_chessboard_l4087_408765

/-- Represents the area of an 8x8 chessboard with one corner square removed -/
def remaining_area : ℕ := 63

/-- Represents the maximum possible area of a single triangle that can fit in the corner -/
def max_triangle_area : ℚ := 7/2

/-- The minimum number of congruent triangles needed to cover the remaining area -/
def min_triangles : ℕ := 18

/-- Theorem stating that the minimum number of congruent triangles needed to cover
    the remaining area of the chessboard is 18 -/
theorem min_triangles_cover_chessboard :
  (remaining_area : ℚ) / max_triangle_area = min_triangles := by sorry

end NUMINAMATH_CALUDE_min_triangles_cover_chessboard_l4087_408765


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4087_408713

theorem simplify_polynomial (z : ℝ) : (4 - 5*z) - (2 + 7*z - z^2) = z^2 - 12*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4087_408713


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l4087_408763

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l4087_408763


namespace NUMINAMATH_CALUDE_visitors_yesterday_l4087_408751

def total_visitors : ℕ := 829
def visitors_today : ℕ := 784

theorem visitors_yesterday (total : ℕ) (today : ℕ) (h1 : total = total_visitors) (h2 : today = visitors_today) :
  total - today = 45 := by
  sorry

end NUMINAMATH_CALUDE_visitors_yesterday_l4087_408751


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l4087_408707

theorem complex_multiplication_simplification :
  let z₁ : ℂ := 5 + 3 * Complex.I
  let z₂ : ℂ := -2 - 6 * Complex.I
  let z₃ : ℂ := 1 - 2 * Complex.I
  (z₁ - z₂) * z₃ = 25 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l4087_408707


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l4087_408710

/-- The interval between segments in systematic sampling -/
def systematic_sampling_interval (N : ℕ) (n : ℕ) : ℕ :=
  N / n

/-- Theorem: For a population of 1500 and a sample size of 50, 
    the systematic sampling interval is 30 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1500 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l4087_408710


namespace NUMINAMATH_CALUDE_equation_satisfied_l4087_408793

theorem equation_satisfied (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 2 * x + 3 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l4087_408793


namespace NUMINAMATH_CALUDE_tan_1500_deg_l4087_408723

theorem tan_1500_deg (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_1500_deg_l4087_408723


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l4087_408726

theorem smallest_divisible_number (n : ℕ) : 
  n = 1008 → 
  (1020 - 12 = n) → 
  (∃ k : ℕ, 36 * k = n) → 
  (∃ k : ℕ, 48 * k = n) → 
  (∃ k : ℕ, 56 * k = n) → 
  ∀ m : ℕ, m ∣ 1008 ∧ m ∣ 36 ∧ m ∣ 48 ∧ m ∣ 56 → m ≤ n :=
by sorry

#check smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_l4087_408726


namespace NUMINAMATH_CALUDE_center_sum_l4087_408794

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 18*y + 9

-- Define the center of the circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k - 9)

-- Theorem statement
theorem center_sum : ∃ h k, is_center h k ∧ h + k = 12 :=
sorry

end NUMINAMATH_CALUDE_center_sum_l4087_408794


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l4087_408770

/-- The area inside a circle but outside a square, when both share the same center -/
theorem area_circle_outside_square (r : ℝ) (s : ℝ) (h : r = Real.sqrt 3 / 3) (hs : s = 2) :
  π * r^2 = π / 3 :=
sorry

end NUMINAMATH_CALUDE_area_circle_outside_square_l4087_408770


namespace NUMINAMATH_CALUDE_farey_consecutive_fraction_l4087_408764

/-- Represents a fraction as a pair of integers -/
structure Fraction where
  numerator : ℤ
  denominator : ℤ
  den_nonzero : denominator ≠ 0

/-- Checks if three fractions are consecutive in a Farey sequence -/
def consecutive_in_farey (f1 f2 f3 : Fraction) : Prop :=
  f1.numerator * f2.denominator - f1.denominator * f2.numerator = 1 ∧
  f3.numerator * f2.denominator - f3.denominator * f2.numerator = 1

/-- The main theorem about three consecutive fractions in a Farey sequence -/
theorem farey_consecutive_fraction (a b c d x y : ℤ) 
  (hb : b ≠ 0) (hd : d ≠ 0) (hy : y ≠ 0)
  (h_order : (a : ℚ) / b < x / y ∧ x / y < c / d)
  (h_consecutive : consecutive_in_farey 
    ⟨a, b, hb⟩ 
    ⟨x, y, hy⟩ 
    ⟨c, d, hd⟩) :
  (x : ℚ) / y = (a + c) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_farey_consecutive_fraction_l4087_408764


namespace NUMINAMATH_CALUDE_root_relation_implies_k_value_l4087_408745

theorem root_relation_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧
   (r+3)^2 - k*(r+3) + 8 = 0 ∧ (s+3)^2 - k*(s+3) + 8 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_relation_implies_k_value_l4087_408745


namespace NUMINAMATH_CALUDE_roses_in_vase_l4087_408741

/-- The total number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: The total number of roses is 22 when there were initially 6 roses and 16 were added -/
theorem roses_in_vase : total_roses 6 16 = 22 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l4087_408741


namespace NUMINAMATH_CALUDE_benny_initial_books_l4087_408732

/-- The number of books Benny had initially -/
def benny_initial : ℕ := sorry

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Sandy received from Benny -/
def sandy_received : ℕ := 10

/-- The total number of books they have together now -/
def total_books : ℕ := 47

theorem benny_initial_books : 
  benny_initial = 24 := by sorry

end NUMINAMATH_CALUDE_benny_initial_books_l4087_408732


namespace NUMINAMATH_CALUDE_total_capacity_l4087_408780

/-- Represents the capacity of boats -/
structure BoatCapacity where
  large : ℕ
  small : ℕ

/-- The capacity of different combinations of boats -/
def boat_combinations (c : BoatCapacity) : Prop :=
  c.large + 4 * c.small = 46 ∧ 2 * c.large + 3 * c.small = 57

/-- The theorem to prove -/
theorem total_capacity (c : BoatCapacity) :
  boat_combinations c → 3 * c.large + 6 * c.small = 96 := by
  sorry


end NUMINAMATH_CALUDE_total_capacity_l4087_408780


namespace NUMINAMATH_CALUDE_translated_function_and_triangle_area_l4087_408719

/-- A linear function f(x) = 3x + b passing through (1, 4) -/
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

theorem translated_function_and_triangle_area (b : ℝ) :
  f b 1 = 4 →
  b = 1 ∧
  (1 / 2 : ℝ) * (1 / 3) * 1 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_translated_function_and_triangle_area_l4087_408719


namespace NUMINAMATH_CALUDE_sum_remainder_l4087_408703

theorem sum_remainder (S : ℤ) : S = (2 * 3^500) / 3 → S % 1000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l4087_408703


namespace NUMINAMATH_CALUDE_divisible_by_eight_l4087_408766

theorem divisible_by_eight (n : ℕ) : ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8*m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l4087_408766


namespace NUMINAMATH_CALUDE_four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l4087_408722

/-- The count of four-digit numbers with no repeated digits -/
def fourDigitNoRepeat : Nat :=
  5 * 4 * 3 * 2

/-- The count of five-digit numbers with no repeated digits and divisible by 5 -/
def fiveDigitNoRepeatDivBy5 : Nat :=
  2 * (4 * 4 * 3 * 2)

theorem four_digit_no_repeat_count :
  fourDigitNoRepeat = 120 := by sorry

theorem five_digit_no_repeat_div_by_5_count :
  fiveDigitNoRepeatDivBy5 = 216 := by sorry

end NUMINAMATH_CALUDE_four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l4087_408722


namespace NUMINAMATH_CALUDE_cos_2x_plus_pi_third_equiv_sin_2x_shifted_l4087_408775

theorem cos_2x_plus_pi_third_equiv_sin_2x_shifted (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_plus_pi_third_equiv_sin_2x_shifted_l4087_408775


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l4087_408778

theorem arithmetic_expression_equality : (3^2 * 5) + (7 * 4) - (42 / 3) = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l4087_408778


namespace NUMINAMATH_CALUDE_two_point_eight_million_scientific_notation_l4087_408742

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_eight_million_scientific_notation :
  toScientificNotation 2800000 = ScientificNotation.mk 2.8 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_two_point_eight_million_scientific_notation_l4087_408742


namespace NUMINAMATH_CALUDE_factorization_equality_l4087_408760

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4087_408760


namespace NUMINAMATH_CALUDE_expression_evaluation_l4087_408711

theorem expression_evaluation : 
  let x : ℚ := 3
  let y : ℚ := -3
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4087_408711


namespace NUMINAMATH_CALUDE_lawn_width_is_30_l4087_408791

/-- Represents the dimensions and properties of a rectangular lawn with roads --/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the total area of the roads on the lawn --/
def road_area (l : LawnWithRoads) : ℝ :=
  l.length * l.road_width + (l.width - l.road_width) * l.road_width

/-- Theorem stating that the width of the lawn is 30 meters --/
theorem lawn_width_is_30 (l : LawnWithRoads) 
  (h1 : l.length = 70)
  (h2 : l.road_width = 5)
  (h3 : l.gravel_cost_per_sqm = 4)
  (h4 : l.total_gravel_cost = 1900)
  : l.width = 30 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_is_30_l4087_408791


namespace NUMINAMATH_CALUDE_f_extrema_l4087_408788

-- Define the function f
def f (x y : ℝ) : ℝ := x^3 + y^3 + 6*x*y

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | -3 ≤ p.1 ∧ p.1 ≤ 1 ∧ -3 ≤ p.2 ∧ p.2 ≤ 2}

theorem f_extrema :
  ∃ (min_point max_point : ℝ × ℝ),
    min_point ∈ rectangle ∧
    max_point ∈ rectangle ∧
    (∀ p ∈ rectangle, f min_point.1 min_point.2 ≤ f p.1 p.2) ∧
    (∀ p ∈ rectangle, f p.1 p.2 ≤ f max_point.1 max_point.2) ∧
    min_point = (-3, 2) ∧
    max_point = (1, 2) ∧
    f min_point.1 min_point.2 = -55 ∧
    f max_point.1 max_point.2 = 21 :=
  sorry


end NUMINAMATH_CALUDE_f_extrema_l4087_408788


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l4087_408739

theorem consecutive_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2) * (k + 3)
  (∃ m : ℤ, n = 11 * m) → 
  (∃ m : ℤ, n = 44 * m) ∧ 
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) * (k + 3) = 66 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l4087_408739


namespace NUMINAMATH_CALUDE_factorization_proof_l4087_408727

theorem factorization_proof (x : ℝ) : (x^2 + 4)^2 - 16*x^2 = (x + 2)^2 * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4087_408727


namespace NUMINAMATH_CALUDE_set_properties_l4087_408709

universe u

variable {U : Type u}
variable (A B C : Set U)

theorem set_properties :
  ((A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C)) ∧
  (Cᶜᶜ = A) :=
by sorry

end NUMINAMATH_CALUDE_set_properties_l4087_408709


namespace NUMINAMATH_CALUDE_max_profit_at_zero_optimal_investment_l4087_408781

/-- Profit function --/
def profit (m : ℝ) : ℝ := 28 - 3 * m

/-- Theorem: The profit function achieves its maximum when m = 0, given m ≥ 0 --/
theorem max_profit_at_zero (m : ℝ) (h : m ≥ 0) : profit 0 ≥ profit m := by
  sorry

/-- Corollary: The optimal investment for maximum profit is 0 --/
theorem optimal_investment : ∃ (m : ℝ), m = 0 ∧ ∀ (n : ℝ), n ≥ 0 → profit m ≥ profit n := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_zero_optimal_investment_l4087_408781


namespace NUMINAMATH_CALUDE_tangent_line_and_zeros_l4087_408744

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 6*x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 6

-- Define the function g
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := f a x - m

theorem tangent_line_and_zeros (a : ℝ) :
  f' a 1 = -6 →
  (∃ b c : ℝ, ∀ x y : ℝ, 12*x + 2*y - 1 = 0 ↔ y = (f a 1) + f' a 1 * (x - 1)) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc (-2) 4 ∧ x₂ ∈ Set.Icc (-2) 4 ∧ x₃ ∈ Set.Icc (-2) 4 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    g a m x₁ = 0 ∧ g a m x₂ = 0 ∧ g a m x₃ = 0) →
    m ∈ Set.Icc (-1) (9/2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_zeros_l4087_408744


namespace NUMINAMATH_CALUDE_wage_increase_l4087_408752

/-- Represents the regression equation for monthly wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 60 * x

/-- Theorem stating that an increase of 1 in labor productivity results in a 60 yuan increase in monthly wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) = wage_equation x + 60 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_l4087_408752


namespace NUMINAMATH_CALUDE_ninth_day_skating_time_l4087_408750

def minutes_per_hour : ℕ := 60

def skating_time_first_5_days : ℕ := 75
def skating_time_next_3_days : ℕ := 90
def total_days : ℕ := 9
def target_average : ℕ := 85

def total_skating_time : ℕ := 
  (skating_time_first_5_days * 5) + (skating_time_next_3_days * 3)

theorem ninth_day_skating_time :
  (total_skating_time + 120) / total_days = target_average :=
sorry

end NUMINAMATH_CALUDE_ninth_day_skating_time_l4087_408750


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l4087_408753

theorem complex_modulus_equality (t : ℝ) (ht : t > 0) : 
  t = 3 * Real.sqrt 3 ↔ Complex.abs (-5 + t * Complex.I) = 2 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l4087_408753


namespace NUMINAMATH_CALUDE_math_competition_proof_l4087_408738

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score : ℕ := 2 * sammy_score
  let cher_score : ℕ := 2 * gab_score
  let alex_score : ℕ := cher_score + (cher_score / 10)
  let combined_score : ℕ := sammy_score + gab_score + cher_score + alex_score
  combined_score - opponent_score = 143

theorem math_competition_proof :
  math_competition 20 85 := by sorry

end NUMINAMATH_CALUDE_math_competition_proof_l4087_408738


namespace NUMINAMATH_CALUDE_equal_benefit_credit_debit_l4087_408705

/-- Represents the benefit of using a card for a purchase -/
structure CardBenefit where
  purchase_amount : ℝ
  cashback_rate : ℝ
  interest_rate : ℝ

/-- Calculates the net benefit of using a card after one month -/
def net_benefit (card : CardBenefit) : ℝ :=
  card.purchase_amount * card.cashback_rate + card.purchase_amount * card.interest_rate

/-- The purchase amount in rubles -/
def purchase_amount : ℝ := 10000

/-- Theorem stating that the net benefit is equal for both credit and debit cards -/
theorem equal_benefit_credit_debit :
  let credit_card := CardBenefit.mk purchase_amount 0.005 0.005
  let debit_card := CardBenefit.mk purchase_amount 0.01 0
  net_benefit credit_card = net_benefit debit_card :=
by sorry

end NUMINAMATH_CALUDE_equal_benefit_credit_debit_l4087_408705


namespace NUMINAMATH_CALUDE_inverse_sum_modulo_thirteen_l4087_408735

theorem inverse_sum_modulo_thirteen : 
  (((5⁻¹ : ZMod 13) + (7⁻¹ : ZMod 13) + (9⁻¹ : ZMod 13) + (11⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 11 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_modulo_thirteen_l4087_408735


namespace NUMINAMATH_CALUDE_c_range_l4087_408774

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → (c - 1) * x + 1 < (c - 1) * y + 1

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 - x + c > 0

theorem c_range (c : ℝ) (hp : p c) (hq : q c) : c > 1 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l4087_408774


namespace NUMINAMATH_CALUDE_problem_statement_l4087_408756

-- Definition of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement :
  (∀ (f : ℝ → ℝ), IsEven (fun x ↦ f x + f (-x))) ∧
  (∀ (f : ℝ → ℝ), IsOdd f → IsOdd (fun x ↦ f (x + 2)) → IsPeriodic f 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4087_408756


namespace NUMINAMATH_CALUDE_grid_paths_count_l4087_408767

/-- The number of paths from (0, 0) to (n, n) on an n × n grid,
    moving only 1 up or 1 right at a time -/
def gridPaths (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

/-- Theorem stating that the number of paths on an n × n grid
    from (0, 0) to (n, n), moving only 1 up or 1 right at a time,
    is equal to (2n choose n) -/
theorem grid_paths_count (n : ℕ) :
  gridPaths n = Nat.choose (2 * n) n := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_count_l4087_408767


namespace NUMINAMATH_CALUDE_f_3_equals_neg_26_l4087_408736

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_3_equals_neg_26 (a b : ℝ) :
  f a b (-3) = 10 → f a b 3 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_neg_26_l4087_408736


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l4087_408762

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l4087_408762


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l4087_408721

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l4087_408721


namespace NUMINAMATH_CALUDE_volumes_equal_l4087_408716

/-- The volume of a solid of revolution obtained by rotating a region about the y-axis -/
noncomputable def VolumeOfRevolution (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def Region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region consisting of points (x, y) that satisfy x²y ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def Region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 * p.2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The theorem stating that the volumes of revolution of the two regions are equal -/
theorem volumes_equal : VolumeOfRevolution Region1 = VolumeOfRevolution Region2 := by
  sorry

end NUMINAMATH_CALUDE_volumes_equal_l4087_408716


namespace NUMINAMATH_CALUDE_divisibility_quotient_l4087_408782

theorem divisibility_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_div : (a * b) ∣ (a^2 + b^2 + 1)) : 
  (a^2 + b^2 + 1) / (a * b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_quotient_l4087_408782


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_range_l4087_408743

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 8 ≥ 0

-- Theorem stating the range of x for which √(x-8) is meaningful
theorem sqrt_x_minus_8_range (x : ℝ) : 
  meaningful_sqrt x ↔ x ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_range_l4087_408743


namespace NUMINAMATH_CALUDE_eggs_left_l4087_408704

/-- Given a box with 47 eggs, if Harry takes 5 eggs and Susan takes x eggs,
    then the number of eggs left in the box is equal to 42 - x. -/
theorem eggs_left (x : ℕ) : 47 - 5 - x = 42 - x := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_l4087_408704


namespace NUMINAMATH_CALUDE_sqrt_inequality_l4087_408799

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1/2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l4087_408799


namespace NUMINAMATH_CALUDE_infiniteSum_eq_one_l4087_408757

/-- Sequence F defined recursively -/
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

/-- The sum of 1/F(2^n) from n=0 to infinity -/
noncomputable def infiniteSum : ℚ := ∑' n, 1 / F (2^n)

/-- Theorem stating that the infinite sum is equal to 1 -/
theorem infiniteSum_eq_one : infiniteSum = 1 := by sorry

end NUMINAMATH_CALUDE_infiniteSum_eq_one_l4087_408757


namespace NUMINAMATH_CALUDE_complex_modulus_l4087_408708

theorem complex_modulus (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I * (Real.sqrt 3)) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4087_408708


namespace NUMINAMATH_CALUDE_sum_of_digits_A_l4087_408769

def A (n : ℕ) : ℕ := 
  match n with
  | 0 => 9
  | m + 1 => A m * (10^(2^(m+1)) - 1)

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem sum_of_digits_A (n : ℕ) : sumOfDigits (A n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_A_l4087_408769


namespace NUMINAMATH_CALUDE_water_flow_proof_l4087_408783

theorem water_flow_proof (rate_second : ℝ) (total_flow : ℝ) : 
  rate_second = 36 →
  ∃ (rate_first rate_third : ℝ),
    rate_second = rate_first * 1.5 ∧
    rate_third = rate_second * 1.25 ∧
    total_flow = rate_first + rate_second + rate_third ∧
    total_flow = 105 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_proof_l4087_408783


namespace NUMINAMATH_CALUDE_perimeter_bounds_l4087_408759

/-- A unit square with 100 segments drawn from its center to its sides, 
    dividing it into 100 parts of equal perimeter -/
structure SegmentedSquare where
  /-- The perimeter of each part -/
  p : ℝ
  /-- The square is a unit square -/
  is_unit_square : True
  /-- There are 100 segments -/
  segment_count : Nat
  segment_count_eq : segment_count = 100
  /-- The square is divided into 100 parts -/
  part_count : Nat
  part_count_eq : part_count = 100
  /-- All parts have equal perimeter -/
  equal_perimeter : True

/-- The perimeter of each part in a segmented unit square satisfies 14/10 < p < 15/10 -/
theorem perimeter_bounds (s : SegmentedSquare) : 14/10 < s.p ∧ s.p < 15/10 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_bounds_l4087_408759


namespace NUMINAMATH_CALUDE_vasya_numbers_l4087_408792

theorem vasya_numbers : 
  ∃ (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_vasya_numbers_l4087_408792


namespace NUMINAMATH_CALUDE_range_of_m_l4087_408730

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / x + 4 / y = 1) → 
  (∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 4 / y = 1 ∧ x + y / 4 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4087_408730


namespace NUMINAMATH_CALUDE_water_displacement_cubed_l4087_408702

/-- Given a cylindrical tank and a partially submerged cube, calculate the volume of water displaced cubed. -/
theorem water_displacement_cubed (tank_radius : ℝ) (cube_side : ℝ) (h : tank_radius = 3 ∧ cube_side = 6) : 
  let submerged_height := cube_side / 2
  let tank_diameter := 2 * tank_radius
  let inscribed_square_side := tank_diameter / Real.sqrt 2
  let intersection_area := inscribed_square_side ^ 2
  let displaced_volume := intersection_area * submerged_height
  displaced_volume ^ 3 = 157464 := by
  sorry

end NUMINAMATH_CALUDE_water_displacement_cubed_l4087_408702


namespace NUMINAMATH_CALUDE_max_value_problem_l4087_408758

theorem max_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) ≤ 3/2 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
    (a^2 * b) / (a + b) + (b^2 * c) / (b + c) + (c^2 * a) / (c + a) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l4087_408758


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l4087_408789

/-- The area of a circle with diameter 8 meters is 16π square meters. -/
theorem circle_area_with_diameter_8 :
  ∃ (A : ℝ), A = π * 16 ∧ A = (π * (8 / 2)^2) := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l4087_408789


namespace NUMINAMATH_CALUDE_fraction_equality_l4087_408734

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4087_408734


namespace NUMINAMATH_CALUDE_lauras_change_l4087_408749

def pants_quantity : ℕ := 2
def pants_price : ℕ := 54
def shirts_quantity : ℕ := 4
def shirts_price : ℕ := 33
def amount_given : ℕ := 250

def total_cost : ℕ := pants_quantity * pants_price + shirts_quantity * shirts_price

theorem lauras_change :
  amount_given - total_cost = 10 := by sorry

end NUMINAMATH_CALUDE_lauras_change_l4087_408749


namespace NUMINAMATH_CALUDE_gala_arrangement_count_l4087_408737

/-- The number of programs in the New Year's gala. -/
def total_programs : ℕ := 8

/-- The number of non-singing programs in the New Year's gala. -/
def non_singing_programs : ℕ := 3

/-- The number of singing programs in the New Year's gala. -/
def singing_programs : ℕ := total_programs - non_singing_programs

/-- A function that calculates the number of ways to arrange the programs
    such that non-singing programs are not adjacent and the first and last
    programs are singing programs. -/
def arrangement_count : ℕ :=
  Nat.choose (total_programs - 2) non_singing_programs *
  Nat.factorial non_singing_programs *
  Nat.factorial (singing_programs - 2)

/-- Theorem stating that the number of ways to arrange the programs
    under the given conditions is 720. -/
theorem gala_arrangement_count :
  arrangement_count = 720 :=
by sorry

end NUMINAMATH_CALUDE_gala_arrangement_count_l4087_408737


namespace NUMINAMATH_CALUDE_minutes_on_eleventh_day_l4087_408725

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 80

/-- The number of minutes Gage skated each day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 95

/-- The total number of days Gage has skated -/
def total_days : ℕ := 10

/-- The desired average number of minutes per day -/
def desired_average : ℕ := 90

/-- The total number of days including the day in question -/
def total_days_with_extra : ℕ := total_days + 1

/-- Theorem stating the number of minutes Gage must skate on the eleventh day -/
theorem minutes_on_eleventh_day :
  (total_days_with_extra * desired_average) - (6 * minutes_per_day_first_6 + 4 * minutes_per_day_next_4) = 130 := by
  sorry

end NUMINAMATH_CALUDE_minutes_on_eleventh_day_l4087_408725


namespace NUMINAMATH_CALUDE_equality_of_fractions_l4087_408706

theorem equality_of_fractions (x y z k : ℝ) 
  (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l4087_408706


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4087_408748

theorem inequality_equivalence (x : ℝ) : 
  (2*x + 1) / 3 ≤ (5*x - 1) / 2 - 1 ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4087_408748


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l4087_408797

/-- Given the cost price and selling price of an article, calculate the profit percentage. -/
theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 500 → selling_price = 675 → 
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l4087_408797


namespace NUMINAMATH_CALUDE_lev_number_pairs_l4087_408720

theorem lev_number_pairs : 
  ∀ a b : ℕ, a + b + a * b = 1000 → 
  ((a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
   (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
   (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_lev_number_pairs_l4087_408720


namespace NUMINAMATH_CALUDE_sqrt_sum_sin_equals_sqrt_two_minus_cos_l4087_408790

theorem sqrt_sum_sin_equals_sqrt_two_minus_cos (α : Real) 
  (h : 5 * Real.pi / 2 ≤ α ∧ α ≤ 7 * Real.pi / 2) : 
  Real.sqrt (1 + Real.sin α) + Real.sqrt (1 - Real.sin α) = Real.sqrt (2 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_sin_equals_sqrt_two_minus_cos_l4087_408790


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_sqrt_8_l4087_408701

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the largest inscribed circle radius for the given quadrilateral is 2√8 -/
theorem largest_inscribed_circle_radius_is_2_sqrt_8 :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 2 * Real.sqrt 8 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_sqrt_8_l4087_408701


namespace NUMINAMATH_CALUDE_singers_and_dancers_selection_l4087_408740

/-- Represents the number of ways to select singers and dancers from a group -/
def select_singers_and_dancers (total : ℕ) (singers : ℕ) (dancers : ℕ) : ℕ :=
  let both := singers + dancers - total
  let only_singers := singers - both
  let only_dancers := dancers - both
  (only_singers * only_dancers) +
  (both * (only_singers + only_dancers)) +
  (both * (both - 1))

/-- Theorem stating that for 9 people with 7 singers and 5 dancers, 
    there are 32 ways to select 2 people and assign one to sing and one to dance -/
theorem singers_and_dancers_selection :
  select_singers_and_dancers 9 7 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_singers_and_dancers_selection_l4087_408740


namespace NUMINAMATH_CALUDE_closest_integer_to_6_sqrt_35_l4087_408700

theorem closest_integer_to_6_sqrt_35 : 
  ∃ n : ℤ, ∀ m : ℤ, |6 * Real.sqrt 35 - n| ≤ |6 * Real.sqrt 35 - m| ∧ n = 36 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_6_sqrt_35_l4087_408700


namespace NUMINAMATH_CALUDE_max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l4087_408798

/-- Given a right-angled triangle with legs a and b, the maximum area of a rectangle
    that can be cut from it, using the right angle of the triangle as one of the
    rectangle's corners, is (a * b) / 4 -/
theorem max_rectangle_area_in_right_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let triangle_area := a * b / 2
  let max_rectangle_area := triangle_area / 2
  max_rectangle_area = a * b / 4 := by sorry

/-- The maximum area of a rectangle that can be cut from a right-angled triangle
    with legs measuring 40 cm and 60 cm, using the right angle of the triangle as
    one of the rectangle's corners, is 600 cm² -/
theorem max_rectangle_area_40_60 :
  let a : ℝ := 40
  let b : ℝ := 60
  let max_area := a * b / 4
  max_area = 600 := by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l4087_408798


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4087_408717

theorem min_value_quadratic :
  ∃ (z_min : ℝ), z_min = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ z_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4087_408717


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l4087_408718

theorem greatest_integer_inequality : 
  (∀ x : ℤ, (1 / 4 : ℚ) + (x : ℚ) / 9 < 7 / 8 → x ≤ 5) ∧ 
  ((1 / 4 : ℚ) + (5 : ℚ) / 9 < 7 / 8) := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l4087_408718
