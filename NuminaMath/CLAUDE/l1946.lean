import Mathlib

namespace NUMINAMATH_CALUDE_cube_root_equals_square_root_l1946_194656

theorem cube_root_equals_square_root :
  ∀ x : ℝ, (x ^ (1/3) = x ^ (1/2)) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equals_square_root_l1946_194656


namespace NUMINAMATH_CALUDE_cake_eaten_after_six_trips_l1946_194635

/-- The fraction of cake eaten after n trips, given that 1/3 is eaten on the first trip
    and half of the remaining cake is eaten on each subsequent trip -/
def cakeEaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else 1/3 + (1 - 1/3) * (1 - (1/2)^(n-1))

/-- The theorem stating that after 6 trips, 47/48 of the cake is eaten -/
theorem cake_eaten_after_six_trips :
  cakeEaten 6 = 47/48 := by sorry

end NUMINAMATH_CALUDE_cake_eaten_after_six_trips_l1946_194635


namespace NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1946_194637

theorem quadratic_roots_always_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1946_194637


namespace NUMINAMATH_CALUDE_lynne_total_spent_l1946_194620

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) (book_cost : ℕ) (magazine_cost : ℕ) : ℕ :=
  (cat_books + solar_books) * book_cost + magazines * magazine_cost

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_total_spent :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

#eval total_spent 7 2 3 7 4

end NUMINAMATH_CALUDE_lynne_total_spent_l1946_194620


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l1946_194631

/-- A type representing questions that can be asked -/
inductive Question
| NumberOfQuestions : Question
| CurrentTime : Question

/-- A type representing the state of the world at a given moment -/
structure WorldState where
  questionsAsked : Nat
  currentTime : Nat

/-- A function that gives the truthful answer to a question given the world state -/
def truthfulAnswer (q : Question) (w : WorldState) : Nat :=
  match q with
  | Question.NumberOfQuestions => w.questionsAsked
  | Question.CurrentTime => w.currentTime

/-- Theorem stating that there exists a question that can have different truthful answers at different times -/
theorem exists_question_with_different_answers :
  ∃ (q : Question) (w1 w2 : WorldState), w1 ≠ w2 → truthfulAnswer q w1 ≠ truthfulAnswer q w2 := by
  sorry


end NUMINAMATH_CALUDE_exists_question_with_different_answers_l1946_194631


namespace NUMINAMATH_CALUDE_joan_piano_time_l1946_194674

/-- Represents the time Joan spent on various activities during her music practice -/
structure MusicPractice where
  total_time : ℕ
  writing_time : ℕ
  reading_time : ℕ
  exercising_time : ℕ

/-- Calculates the time spent on the piano given Joan's music practice schedule -/
def time_on_piano (practice : MusicPractice) : ℕ :=
  practice.total_time - (practice.writing_time + practice.reading_time + practice.exercising_time)

/-- Theorem stating that Joan spent 30 minutes on the piano -/
theorem joan_piano_time :
  let practice : MusicPractice := {
    total_time := 120,
    writing_time := 25,
    reading_time := 38,
    exercising_time := 27
  }
  time_on_piano practice = 30 := by sorry

end NUMINAMATH_CALUDE_joan_piano_time_l1946_194674


namespace NUMINAMATH_CALUDE_area_ACE_is_60_l1946_194641

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the intersection point O of diagonals AC and BD
def O (q : Quadrilateral) : Point := sorry

-- Define the height DE of triangle DBC
def DE (q : Quadrilateral) : Real := 15

-- Define the length of DC
def DC (q : Quadrilateral) : Real := 17

-- Define the areas of triangles
def area_ABO (q : Quadrilateral) : Real := sorry
def area_DCO (q : Quadrilateral) : Real := sorry
def area_ACE (q : Quadrilateral) : Real := sorry

-- State the theorem
theorem area_ACE_is_60 (q : Quadrilateral) :
  area_ABO q = area_DCO q → area_ACE q = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_ACE_is_60_l1946_194641


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l1946_194630

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_specific_gp :
  let a := Real.sqrt 2
  let r := (Real.sqrt 2) ^ (1/4)
  let third_term := geometric_progression a r 3
  third_term = 2 ^ (1/8) →
  geometric_progression a r 4 = 1 / (Real.sqrt 2) ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l1946_194630


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l1946_194643

theorem number_of_elements_in_set (initial_average : ℚ) (incorrect_number : ℚ) (correct_number : ℚ) (correct_average : ℚ) (n : ℕ) : 
  initial_average = 21 →
  incorrect_number = 26 →
  correct_number = 36 →
  correct_average = 22 →
  (n : ℚ) * initial_average + (correct_number - incorrect_number) = (n : ℚ) * correct_average →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l1946_194643


namespace NUMINAMATH_CALUDE_circle_intersection_tangent_slope_l1946_194661

noncomputable def C₁ (x y : ℝ) := x^2 + y^2 - 6*x + 4*y + 9 = 0

noncomputable def C₂ (m x y : ℝ) := (x + m)^2 + (y + m + 5)^2 = 2*m^2 + 8*m + 10

def on_coordinate_axes (x y : ℝ) := x = 0 ∨ y = 0

theorem circle_intersection_tangent_slope 
  (m : ℝ) (h_m : m ≠ -3) (x₀ y₀ : ℝ) (h_axes : on_coordinate_axes x₀ y₀)
  (h_tangent : ∃ (T₁_x T₁_y T₂_x T₂_y : ℝ), 
    C₁ T₁_x T₁_y ∧ C₂ m T₂_x T₂_y ∧ 
    (x₀ - T₁_x)^2 + (y₀ - T₁_y)^2 = (x₀ - T₂_x)^2 + (y₀ - T₂_y)^2) :
  (m = 5 → ∃! (n : ℕ), n = 2 ∧ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ m x₁ y₁ ∧ C₂ m x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ∧
  (x₀ + y₀ + 1 = 0 ∧ ((x₀ = 0 ∧ y₀ = -1) ∨ (x₀ = -1 ∧ y₀ = 0))) ∧
  (∀ (k : ℝ), (∀ (x y : ℝ), C₁ x y → (y + 2 = k * (x - 3)) → 
    (∀ (m : ℝ), m ≠ -3 → ∃ (x' y' : ℝ), C₂ m x' y' ∧ y' + 2 = k * (x' - 3))) → k > 0) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_tangent_slope_l1946_194661


namespace NUMINAMATH_CALUDE_remainder_theorem_l1946_194682

def polynomial (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40
def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-10) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1946_194682


namespace NUMINAMATH_CALUDE_base_difference_theorem_l1946_194647

/-- Convert a number from base 16 to base 10 -/
def base16_to_base10 (n : String) : ℕ :=
  match n with
  | "1A3" => 419
  | _ => 0

/-- Convert a number from base 7 to base 10 -/
def base7_to_base10 (n : String) : ℕ :=
  match n with
  | "142" => 79
  | _ => 0

/-- The main theorem stating that the difference between 1A3 (base 16) and 142 (base 7) in base 10 is 340 -/
theorem base_difference_theorem :
  base16_to_base10 "1A3" - base7_to_base10 "142" = 340 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l1946_194647


namespace NUMINAMATH_CALUDE_steve_bike_time_l1946_194606

/-- Given that Steve biked 5 miles in the same time Jordan biked 3 miles,
    and Jordan took 18 minutes to bike 3 miles,
    prove that Steve will take 126/5 minutes to bike 7 miles. -/
theorem steve_bike_time (steve_distance : ℝ) (jordan_distance : ℝ) (jordan_time : ℝ) (steve_new_distance : ℝ) :
  steve_distance = 5 →
  jordan_distance = 3 →
  jordan_time = 18 →
  steve_new_distance = 7 →
  (steve_new_distance / (steve_distance / jordan_time)) = 126 / 5 := by
  sorry

end NUMINAMATH_CALUDE_steve_bike_time_l1946_194606


namespace NUMINAMATH_CALUDE_even_function_implies_a_plus_b_eq_four_l1946_194636

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function on an interval
def is_even_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x, l ≤ x ∧ x ≤ r → f x = f (-x)

-- Theorem statement
theorem even_function_implies_a_plus_b_eq_four (a b : ℝ) :
  is_even_on (f a b) (a^2 - 2) a →
  a^2 - 2 ≤ a →
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_even_function_implies_a_plus_b_eq_four_l1946_194636


namespace NUMINAMATH_CALUDE_apples_to_sell_for_target_profit_l1946_194653

/-- Represents the number of apples bought in one transaction -/
def apples_bought : ℕ := 4

/-- Represents the cost in cents for buying apples_bought apples -/
def buying_cost : ℕ := 15

/-- Represents the number of apples sold in one transaction -/
def apples_sold : ℕ := 7

/-- Represents the revenue in cents from selling apples_sold apples -/
def selling_revenue : ℕ := 35

/-- Represents the target profit in cents -/
def target_profit : ℕ := 140

/-- Theorem stating that 112 apples need to be sold to achieve the target profit -/
theorem apples_to_sell_for_target_profit :
  (selling_revenue * 112 / apples_sold) - (buying_cost * 112 / apples_bought) = target_profit := by
  sorry

end NUMINAMATH_CALUDE_apples_to_sell_for_target_profit_l1946_194653


namespace NUMINAMATH_CALUDE_square_difference_l1946_194665

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1946_194665


namespace NUMINAMATH_CALUDE_divisibility_of_cubic_difference_l1946_194649

theorem divisibility_of_cubic_difference (x a b : ℝ) :
  ∃ P : ℝ → ℝ, (x + a + b)^3 - x^3 - a^3 - b^3 = P x * ((x + a) * (x + b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_cubic_difference_l1946_194649


namespace NUMINAMATH_CALUDE_legacy_gold_bars_l1946_194604

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := sorry

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The value of one gold bar in dollars -/
def bar_value : ℕ := 2200

/-- The total value of gold bars Legacy and Aleena have together -/
def total_value : ℕ := 17600

theorem legacy_gold_bars :
  legacy_bars = 5 ∧
  aleena_bars = legacy_bars - 2 ∧
  bar_value = 2200 ∧
  total_value = 17600 ∧
  total_value = bar_value * (legacy_bars + aleena_bars) :=
sorry

end NUMINAMATH_CALUDE_legacy_gold_bars_l1946_194604


namespace NUMINAMATH_CALUDE_range_of_trigonometric_function_l1946_194627

theorem range_of_trigonometric_function :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 →
  (π / 2 - Real.arctan 2) ≤ Real.arcsin x + Real.arccos x + Real.arctan (2 * x) ∧
  Real.arcsin x + Real.arccos x + Real.arctan (2 * x) ≤ (π / 2 + Real.arctan 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_trigonometric_function_l1946_194627


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1946_194642

def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 16

def is_on_circle (x y : ℝ) : Prop :=
  ((x - 4)^2 + (y + 6)^2) = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, is_on_circle x y ↔ 
    ((x - 4)^2 + (y + 6)^2 = 16 ∧ 
     (x - 4)^2 + (y - (-6))^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1946_194642


namespace NUMINAMATH_CALUDE_painted_cubes_l1946_194621

theorem painted_cubes (n : ℕ) (h : n = 4) :
  let total_cubes := n^3
  let unpainted_cubes := (n - 2)^3
  let painted_cubes := total_cubes - unpainted_cubes
  painted_cubes = 42 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l1946_194621


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_line_l1946_194611

/-- Given a line 3x - 4y + 12 = 0 intersecting the x-axis and y-axis, 
    the circle with the line segment between these intersections as its diameter 
    has the equation x^2 + 4x + y^2 - 3y = 0 -/
theorem circle_equation_from_diameter_line (x y : ℝ) : 
  (∃ (t : ℝ), 3*t - 4*0 + 12 = 0 ∧ 3*0 - 4*t + 12 = 0) →  -- Line intersects both axes
  (x^2 + 4*x + y^2 - 3*y = 0 ↔ 
   ∃ (p : ℝ × ℝ), (3*(p.1) - 4*(p.2) + 12 = 0) ∧ 
                  ((x - p.1)^2 + (y - p.2)^2 = 
                   ((3*0 - 4*0 + 12)/3 - (3*0 - 4*0 + 12)/(-4))^2/4 + 
                   ((3*0 - 4*0 + 12)/(-4))^2/4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_line_l1946_194611


namespace NUMINAMATH_CALUDE_simplify_expression_l1946_194675

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 9 = 45*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1946_194675


namespace NUMINAMATH_CALUDE_train_length_l1946_194691

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ (length : ℝ), abs (length - 150.03) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l1946_194691


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1946_194671

/-- The area of a circle with diameter endpoints C(-2, 3) and D(4, -1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1946_194671


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1946_194623

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Calculates the total duration of all meetings in minutes -/
def total_meeting_minutes : ℕ := 
  first_meeting_minutes + 2 * first_meeting_minutes + (first_meeting_minutes + 2 * first_meeting_minutes)

/-- Theorem stating that the percentage of work day spent in meetings is 35% -/
theorem meetings_percentage_of_workday : 
  (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1946_194623


namespace NUMINAMATH_CALUDE_abs_neg_two_and_half_l1946_194615

theorem abs_neg_two_and_half : |(-5/2 : ℚ)| = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_and_half_l1946_194615


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l1946_194617

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 1 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l1946_194617


namespace NUMINAMATH_CALUDE_total_savings_percentage_l1946_194692

/-- Calculates the total savings percentage given the original prices and discount rates -/
theorem total_savings_percentage
  (jacket_price shirt_price hat_price : ℝ)
  (jacket_discount shirt_discount hat_discount : ℝ)
  (h_jacket_price : jacket_price = 100)
  (h_shirt_price : shirt_price = 50)
  (h_hat_price : hat_price = 30)
  (h_jacket_discount : jacket_discount = 0.3)
  (h_shirt_discount : shirt_discount = 0.6)
  (h_hat_discount : hat_discount = 0.5) :
  (jacket_price * jacket_discount + shirt_price * shirt_discount + hat_price * hat_discount) /
  (jacket_price + shirt_price + hat_price) * 100 = 41.67 :=
by sorry

end NUMINAMATH_CALUDE_total_savings_percentage_l1946_194692


namespace NUMINAMATH_CALUDE_curve_self_intersection_l1946_194652

theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧
    a^2 - 4 = b^2 - 4 ∧
    a^3 - 6*a + 4 = b^3 - 6*b + 4 ∧
    (a^2 - 4 = 2 ∧ a^3 - 6*a + 4 = 4) :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l1946_194652


namespace NUMINAMATH_CALUDE_box_width_is_ten_inches_l1946_194651

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_width_is_ten_inches (box : Dimensions) (block : Dimensions) :
  box.height = 8 →
  box.length = 12 →
  block.height = 3 →
  block.width = 2 →
  block.length = 4 →
  volume box = 40 * volume block →
  box.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_ten_inches_l1946_194651


namespace NUMINAMATH_CALUDE_decimal_34_to_binary_l1946_194693

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_34_to_binary :
  decimal_to_binary 34 = [1, 0, 0, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_34_to_binary_l1946_194693


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1946_194698

/-- The speed of a train in km/hr -/
def train_speed : ℝ := 90

/-- The length of the train in meters -/
def train_length : ℝ := 750

/-- The time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- The length of the platform in meters -/
def platform_length : ℝ := train_length

theorem train_speed_calculation :
  train_speed = (2 * train_length) / crossing_time * 60 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1946_194698


namespace NUMINAMATH_CALUDE_sample_capacity_l1946_194625

theorem sample_capacity (f : ℕ) (fr : ℚ) (h1 : f = 36) (h2 : fr = 1/4) :
  ∃ n : ℕ, f / n = fr ∧ n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l1946_194625


namespace NUMINAMATH_CALUDE_infinitely_many_n_divides_2_pow_n_plus_2_l1946_194687

theorem infinitely_many_n_divides_2_pow_n_plus_2 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n > 0 ∧ n ∣ 2^n + 2 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_divides_2_pow_n_plus_2_l1946_194687


namespace NUMINAMATH_CALUDE_count_sequences_100_l1946_194685

/-- The number of sequences of length n, where each sequence contains at least one 4 or 5,
    and any two consecutive members differ by no more than 2. -/
def count_sequences (n : ℕ) : ℕ :=
  5^n - 3^n

/-- The theorem stating that the number of valid sequences of length 100 is 5^100 - 3^100. -/
theorem count_sequences_100 :
  count_sequences 100 = 5^100 - 3^100 :=
by sorry

end NUMINAMATH_CALUDE_count_sequences_100_l1946_194685


namespace NUMINAMATH_CALUDE_cube_equality_condition_l1946_194695

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  (edge_length : n > 3)

/-- The number of unit cubes with exactly two faces painted -/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 4)

/-- The number of unit cubes with no faces painted -/
def no_faces_painted (c : Cube n) : ℕ := (n - 2)^3

/-- Theorem stating the equality condition for n = 5 -/
theorem cube_equality_condition (n : ℕ) (c : Cube n) :
  two_faces_painted c = no_faces_painted c ↔ n = 5 :=
sorry

end NUMINAMATH_CALUDE_cube_equality_condition_l1946_194695


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1946_194670

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * (x - 1) / (x - 2) > 1}
  (a = 1 → solution_set = {x : ℝ | x > 2}) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | (a - 2) / (1 - a) < x ∧ x < 2}) ∧
  (a > 1 → solution_set = {x : ℝ | x < (a - 2) / (a - 1) ∨ x > 2}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1946_194670


namespace NUMINAMATH_CALUDE_max_store_visits_l1946_194605

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : double_visitors = 8) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_store_visits_l1946_194605


namespace NUMINAMATH_CALUDE_blithe_toys_proof_l1946_194676

/-- The number of toys Blithe lost -/
def lost_toys : ℕ := 6

/-- The number of toys Blithe found -/
def found_toys : ℕ := 9

/-- The number of toys Blithe had after losing and finding toys -/
def final_toys : ℕ := 43

/-- The initial number of toys Blithe had -/
def initial_toys : ℕ := 40

theorem blithe_toys_proof :
  initial_toys - lost_toys + found_toys = final_toys :=
by sorry

end NUMINAMATH_CALUDE_blithe_toys_proof_l1946_194676


namespace NUMINAMATH_CALUDE_evaluate_expression_l1946_194614

theorem evaluate_expression (x : ℝ) (y : ℝ) (h1 : x = 5) (h2 : y = 2 * x) : 
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1946_194614


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1946_194668

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ)
  (h1 : sold = 30)
  (h2 : cages = 6)
  (h3 : puppies_per_cage = 8) :
  sold + cages * puppies_per_cage = 78 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1946_194668


namespace NUMINAMATH_CALUDE_largest_monochromatic_subgraph_2024_l1946_194663

/-- A 3-coloring of the edges of a complete graph -/
def ThreeColoring (n : ℕ) := Fin 3 → Sym2 (Fin n)

/-- A function that returns the size of the largest monochromatic connected subgraph -/
noncomputable def largestMonochromaticSubgraph (n : ℕ) (coloring : ThreeColoring n) : ℕ := sorry

theorem largest_monochromatic_subgraph_2024 :
  ∀ (coloring : ThreeColoring 2024),
  largestMonochromaticSubgraph 2024 coloring ≥ 1012 := by sorry

end NUMINAMATH_CALUDE_largest_monochromatic_subgraph_2024_l1946_194663


namespace NUMINAMATH_CALUDE_nine_students_in_front_of_hoseok_l1946_194602

/-- The number of students standing in front of Hoseok in a line of 20 students, 
    where 11 students are behind Yoongi and Hoseok is right behind Yoongi. -/
def studentsInFrontOfHoseok (totalStudents : Nat) (studentsBehinYoongi : Nat) : Nat :=
  totalStudents - studentsBehinYoongi

/-- Theorem stating that 9 students are in front of Hoseok given the conditions -/
theorem nine_students_in_front_of_hoseok :
  studentsInFrontOfHoseok 20 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_students_in_front_of_hoseok_l1946_194602


namespace NUMINAMATH_CALUDE_ring_stack_height_is_117_l1946_194662

/-- Calculates the distance from the top of the top ring to the bottom of the bottom ring in a stack of linked rings. -/
def ring_stack_height (top_diameter : ℝ) (top_thickness : ℝ) (bottom_diameter : ℝ) 
  (diameter_decrease : ℝ) (thickness_decrease : ℝ) : ℝ :=
  sorry

/-- The distance from the top of the top ring to the bottom of the bottom ring is 117 cm. -/
theorem ring_stack_height_is_117 : 
  ring_stack_height 30 2 10 2 0.1 = 117 := by sorry

end NUMINAMATH_CALUDE_ring_stack_height_is_117_l1946_194662


namespace NUMINAMATH_CALUDE_distinct_pair_count_l1946_194689

theorem distinct_pair_count :
  let S := Finset.range 15
  (S.card * (S.card - 1) : ℕ) = 210 := by sorry

end NUMINAMATH_CALUDE_distinct_pair_count_l1946_194689


namespace NUMINAMATH_CALUDE_stating_tournament_winners_l1946_194633

/-- Represents the number of participants in a tournament round -/
def participants : ℕ := 512

/-- Represents the number of wins we're interested in -/
def target_wins : ℕ := 6

/-- 
Represents the number of participants who finish with exactly k wins 
in a single-elimination tournament with n rounds
-/
def participants_with_wins (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that in a single-elimination tournament with 512 participants,
exactly 84 participants will finish with 6 wins
-/
theorem tournament_winners : 
  participants_with_wins (Nat.log 2 participants) target_wins = 84 := by
  sorry

end NUMINAMATH_CALUDE_stating_tournament_winners_l1946_194633


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l1946_194677

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (StudiesLate : U → Prop)

-- State the theorem
theorem some_students_not_club_members
  (h1 : ∃ x, Student x ∧ ¬StudiesLate x)
  (h2 : ∀ x, ClubMember x → StudiesLate x) :
  ∃ x, Student x ∧ ¬ClubMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_students_not_club_members_l1946_194677


namespace NUMINAMATH_CALUDE_triangle_side_length_l1946_194696

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- Area formula
  Real.sin A + Real.sin C = 2 * Real.sin B →  -- Given condition
  b = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1946_194696


namespace NUMINAMATH_CALUDE_absolute_value_and_square_l1946_194607

theorem absolute_value_and_square (x : ℝ) : 
  (x < 0 → abs x > x) ∧ (x > 2 → x^2 > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_square_l1946_194607


namespace NUMINAMATH_CALUDE_nell_gave_28_cards_l1946_194669

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Proof that Nell gave 28 cards to Jeff -/
theorem nell_gave_28_cards :
  cards_given_to_jeff 304 276 = 28 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_28_cards_l1946_194669


namespace NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l1946_194694

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_size_scientific_notation :
  toScientificNotation 0.0000012 = ScientificNotation.mk 1.2 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l1946_194694


namespace NUMINAMATH_CALUDE_knight_return_even_moves_l1946_194690

/-- Represents a chess square --/
structure ChessSquare :=
  (color : Bool)

/-- Represents a knight's move on a chess board --/
def knightMove (start : ChessSquare) : ChessSquare :=
  { color := ¬start.color }

/-- Represents a sequence of knight moves --/
def knightMoves (start : ChessSquare) (n : ℕ) : ChessSquare :=
  match n with
  | 0 => start
  | m + 1 => knightMove (knightMoves start m)

/-- Theorem: If a knight returns to its starting square after n moves, then n is even --/
theorem knight_return_even_moves (start : ChessSquare) (n : ℕ) :
  knightMoves start n = start → Even n :=
by sorry

end NUMINAMATH_CALUDE_knight_return_even_moves_l1946_194690


namespace NUMINAMATH_CALUDE_equation_solutions_l1946_194618

theorem equation_solutions :
  (∀ x : ℚ, 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2) ∧
  (∀ x : ℚ, 2 * x^2 + 3 * x - 5 = 0 ↔ x = -5/2 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1946_194618


namespace NUMINAMATH_CALUDE_problem_statement_l1946_194600

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that c^2 * (a + b) = 2008 -/
theorem problem_statement (a b c : ℝ) 
    (h1 : a^2 * (b + c) = 2008)
    (h2 : b^2 * (a + c) = 2008)
    (h3 : a ≠ b) :
  c^2 * (a + b) = 2008 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1946_194600


namespace NUMINAMATH_CALUDE_meeting_at_163rd_streetlight_l1946_194659

/-- The number of streetlights along the alley -/
def num_streetlights : ℕ := 400

/-- The position where Alla and Boris meet -/
def meeting_point : ℕ := 163

/-- Alla's position when observation is made -/
def alla_observed_pos : ℕ := 55

/-- Boris's position when observation is made -/
def boris_observed_pos : ℕ := 321

/-- The theorem stating that Alla and Boris meet at the 163rd streetlight -/
theorem meeting_at_163rd_streetlight :
  let alla_distance := alla_observed_pos - 1
  let boris_distance := num_streetlights - boris_observed_pos
  let total_observed_distance := alla_distance + boris_distance
  let scaling_factor := (num_streetlights - 1) / total_observed_distance
  (1 : ℚ) + scaling_factor * alla_distance = meeting_point := by
  sorry

end NUMINAMATH_CALUDE_meeting_at_163rd_streetlight_l1946_194659


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1946_194667

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1946_194667


namespace NUMINAMATH_CALUDE_function_and_minimum_value_l1946_194673

def f (x : ℝ) := x^2 - x - 2

def g (x a : ℝ) := f (x + a) + x

theorem function_and_minimum_value :
  (∀ x, f (x - 1) = x^2 - 3 * x) →
  (∀ x, f x = x^2 - x - 2) ∧
  (∀ a,
    (a ≥ 1 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ a^2 - 3 * a - 1) ∧
    (-3 < a ∧ a < 1 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ -a - 2) ∧
    (a ≤ -3 → ∀ x ∈ Set.Icc (-1) 3, g x a ≥ a^2 + 5 * a + 7)) :=
by sorry

end NUMINAMATH_CALUDE_function_and_minimum_value_l1946_194673


namespace NUMINAMATH_CALUDE_circle_through_intersections_and_tangent_to_line_l1946_194628

/-- Given two circles and a line, prove that a specific circle passes through 
    the intersection points of the given circles and is tangent to the given line. -/
theorem circle_through_intersections_and_tangent_to_line :
  let C₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 4
  let C₂ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 - 2*x - 4*y + 4 = 0
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y = 0
  let result_circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1/2)^2 + (y - 1)^2 = 5/4
  
  (∀ p, C₁ p ∧ C₂ p → result_circle p) ∧ 
  (∃ unique_p, l unique_p ∧ result_circle unique_p ∧ 
    ∀ q, l q ∧ result_circle q → q = unique_p) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_through_intersections_and_tangent_to_line_l1946_194628


namespace NUMINAMATH_CALUDE_ella_jasper_passing_count_l1946_194613

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the track in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

/-- Theorem: Ella and Jasper pass each other 93 times during their 40-minute jog -/
theorem ella_jasper_passing_count : 
  let ella : Runner := { speed := 300, radius := 40, direction := 1 }
  let jasper : Runner := { speed := 360, radius := 50, direction := -1 }
  passingCount ella jasper 40 = 93 := by
  sorry

end NUMINAMATH_CALUDE_ella_jasper_passing_count_l1946_194613


namespace NUMINAMATH_CALUDE_max_value_of_s_l1946_194640

theorem max_value_of_s (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  s ≤ Real.sqrt 2 ∧ 
  (s = Real.sqrt 2 ↔ x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_s_l1946_194640


namespace NUMINAMATH_CALUDE_middle_card_number_l1946_194632

theorem middle_card_number (a b c : ℕ) : 
  a < b → b < c → 
  a + b + c = 15 → 
  a + b < 10 → 
  (∀ x y z : ℕ, x < y → y < z → x + y + z = 15 → x + y < 10 → (x = a ∧ z = c) → y ≠ b) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_number_l1946_194632


namespace NUMINAMATH_CALUDE_max_value_of_z_l1946_194699

theorem max_value_of_z (x y : ℝ) 
  (h1 : |2*x + y + 1| ≤ |x + 2*y + 2|) 
  (h2 : -1 ≤ y) (h3 : y ≤ 1) : 
  (∀ (x' y' : ℝ), |2*x' + y' + 1| ≤ |x' + 2*y' + 2| → -1 ≤ y' → y' ≤ 1 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 5 := by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l1946_194699


namespace NUMINAMATH_CALUDE_neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l1946_194638

-- Problem 1
theorem neg_neg_two_eq_two : -(-2) = 2 := by sorry

-- Problem 2
theorem neg_six_plus_six_eq_zero : -6 + 6 = 0 := by sorry

-- Problem 3
theorem neg_three_times_five_eq_neg_fifteen : (-3) * 5 = -15 := by sorry

-- Problem 4
theorem two_x_minus_three_x_eq_neg_x (x : ℤ) : 2*x - 3*x = -x := by sorry

end NUMINAMATH_CALUDE_neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l1946_194638


namespace NUMINAMATH_CALUDE_james_car_value_l1946_194658

/-- The value of James' old car -/
def old_car_value : ℝ := 20000

/-- The percentage of the old car's value James received when selling it -/
def old_car_sell_percentage : ℝ := 0.8

/-- The sticker price of the new car -/
def new_car_sticker_price : ℝ := 30000

/-- The percentage of the new car's sticker price James paid after haggling -/
def new_car_buy_percentage : ℝ := 0.9

/-- The out-of-pocket amount James paid -/
def out_of_pocket : ℝ := 11000

theorem james_car_value :
  new_car_buy_percentage * new_car_sticker_price - old_car_sell_percentage * old_car_value = out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_james_car_value_l1946_194658


namespace NUMINAMATH_CALUDE_go_game_competition_l1946_194664

/-- Represents the probability of a player winning a single game -/
structure GameProbability where
  player_a : ℝ
  player_b : ℝ
  sum_to_one : player_a + player_b = 1

/-- Represents the state of the game after the first two games -/
structure GameState where
  a_wins : ℕ
  b_wins : ℕ
  total_games : a_wins + b_wins = 2

/-- The probability of the competition ending after 2 more games -/
def probability_end_in_two_more_games (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + p.player_b * p.player_b

/-- The probability of player A winning the competition -/
def probability_a_wins (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + 
  p.player_b * p.player_a * p.player_a + 
  p.player_a * p.player_b * p.player_a

theorem go_game_competition 
  (p : GameProbability) 
  (state : GameState) 
  (h_p : p.player_a = 0.6 ∧ p.player_b = 0.4) 
  (h_state : state.a_wins = 1 ∧ state.b_wins = 1) : 
  probability_end_in_two_more_games p = 0.52 ∧ 
  probability_a_wins p = 0.648 := by
  sorry


end NUMINAMATH_CALUDE_go_game_competition_l1946_194664


namespace NUMINAMATH_CALUDE_job_completion_time_l1946_194603

/-- Given two workers A and B who can complete a job in 15 and 30 days respectively,
    prove that they worked together for 4 days if 0.6 of the job is left unfinished. -/
theorem job_completion_time 
  (rate_A : ℝ) (rate_B : ℝ) (days_worked : ℝ) (fraction_left : ℝ) :
  rate_A = 1 / 15 →
  rate_B = 1 / 30 →
  fraction_left = 0.6 →
  (rate_A + rate_B) * days_worked = 1 - fraction_left →
  days_worked = 4 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1946_194603


namespace NUMINAMATH_CALUDE_proportion_equality_l1946_194644

theorem proportion_equality (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1946_194644


namespace NUMINAMATH_CALUDE_two_digit_reverse_sqrt_l1946_194654

theorem two_digit_reverse_sqrt (n x y : ℕ) : 
  (x > y) →
  (2 * n = x + y) →
  (10 ≤ n ∧ n < 100) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10) →
  (∃ (k : ℕ), k * k = x * y) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ k = 10 * b + a) →
  (x - y = 66) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sqrt_l1946_194654


namespace NUMINAMATH_CALUDE_extremum_implies_a_and_monotonicity_l1946_194639

/-- The function f(x) = ax^3 - x^2 with a ∈ ℝ -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x

theorem extremum_implies_a_and_monotonicity (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1/3 ∧ |x - 1/3| < ε → |f a x| ≤ |f a (1/3)|) →
  (a = 6 ∧
   (∀ (x y : ℝ), (x < y ∧ y < 0) → f a x < f a y) ∧
   (∀ (x y : ℝ), (0 < x ∧ x < y ∧ y < 1/3) → f a x > f a y) ∧
   (∀ (x y : ℝ), (1/3 < x ∧ x < y) → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_and_monotonicity_l1946_194639


namespace NUMINAMATH_CALUDE_larger_number_is_eleven_l1946_194686

theorem larger_number_is_eleven (x y : ℝ) (h1 : y - x = 2) (h2 : x + y = 20) : 
  max x y = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_eleven_l1946_194686


namespace NUMINAMATH_CALUDE_geometry_theorem_l1946_194622

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : α ≠ β) 
  (h_distinct_lines : m ≠ n) :
  (∀ (m n : Line) (α : Plane), perpendicular m α → perpendicular n α → parallel m n) ∧
  (∀ (m n : Line) (α β : Plane), perpendicular m α → parallel m n → contains β n → planePerp α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1946_194622


namespace NUMINAMATH_CALUDE_line_direction_vector_l1946_194666

def point_a : ℝ × ℝ := (-3, 1)
def point_b : ℝ × ℝ := (2, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (1, b)

theorem line_direction_vector : 
  ∃ (b : ℝ), (point_b.1 - point_a.1, point_b.2 - point_a.2) = 
    (point_b.1 - point_a.1) • direction_vector b ∧ b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1946_194666


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l1946_194629

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability that a randomly selected point on PQ is between R and S is 5/8 -/
theorem probability_between_R_and_S (P Q R S : ℝ) (h1 : P < R) (h2 : R < S) (h3 : S < Q)
    (h4 : Q - P = 4 * (R - P)) (h5 : Q - P = 8 * (S - R)) :
    (S - R) / (Q - P) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l1946_194629


namespace NUMINAMATH_CALUDE_eyes_seeing_airplane_l1946_194688

/-- Given 200 students on a field and 3/4 of them looking up at an airplane,
    prove that the number of eyes that saw the airplane is 300. -/
theorem eyes_seeing_airplane (total_students : ℕ) (fraction_looking_up : ℚ) : 
  total_students = 200 →
  fraction_looking_up = 3/4 →
  (total_students : ℚ) * fraction_looking_up * 2 = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_eyes_seeing_airplane_l1946_194688


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1946_194697

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (16 - m) + y^2 / (m + 4) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  (16 - m > 0) ∧ (m + 4 > 0) ∧ (16 - m ≠ m + 4)

/-- Theorem stating the range of m for which the equation represents an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (m > -4 ∧ m < 16 ∧ m ≠ 6) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1946_194697


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l1946_194650

theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 30 * Real.pi * r) → h / r = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l1946_194650


namespace NUMINAMATH_CALUDE_largest_number_l1946_194634

theorem largest_number (a b c d : ℝ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l1946_194634


namespace NUMINAMATH_CALUDE_f_is_even_l1946_194646

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g x = f g (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1946_194646


namespace NUMINAMATH_CALUDE_range_of_a_l1946_194648

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, (1/2) * x^2 - Real.log (x - a) ≥ 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_a : Set ℝ := Set.Ici (-4) ∪ Set.Icc (-2) (1/2)

-- Theorem statement
theorem range_of_a (a : ℝ) : p a ∧ q a → a ∈ range_a := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1946_194648


namespace NUMINAMATH_CALUDE_max_value_expression_l1946_194657

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * y * Real.sqrt 5 + 9 * y * z ≤ (3/2) * Real.sqrt 409 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1946_194657


namespace NUMINAMATH_CALUDE_tree_planting_event_l1946_194678

theorem tree_planting_event (boys girls : ℕ) : 
  girls - boys = 400 →
  boys = 600 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls) = 960 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_event_l1946_194678


namespace NUMINAMATH_CALUDE_probability_under_20_l1946_194683

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 2/5 := by sorry

end NUMINAMATH_CALUDE_probability_under_20_l1946_194683


namespace NUMINAMATH_CALUDE_inequality_condition_l1946_194601

theorem inequality_condition (x : ℝ) : (x - Real.pi) * (x - Real.exp 1) ≤ 0 ↔ Real.exp 1 < x ∧ x < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1946_194601


namespace NUMINAMATH_CALUDE_school_commute_theorem_l1946_194609

/-- Represents the number of students in different commute categories -/
structure SchoolCommute where
  localStudents : ℕ
  publicTransport : ℕ
  privateTransport : ℕ
  train : ℕ
  bus : ℕ
  cycle : ℕ
  drivenByParents : ℕ

/-- Given the commute ratios and public transport users, proves the number of train commuters
    minus parent-driven students and the total number of students -/
theorem school_commute_theorem (sc : SchoolCommute) 
  (h1 : sc.localStudents = 3 * (sc.publicTransport + sc.privateTransport))
  (h2 : 3 * sc.privateTransport = 2 * sc.publicTransport)
  (h3 : 7 * sc.bus = 5 * sc.train)
  (h4 : 5 * sc.drivenByParents = 3 * sc.cycle)
  (h5 : sc.publicTransport = 24)
  (h6 : sc.publicTransport = sc.train + sc.bus)
  (h7 : sc.privateTransport = sc.cycle + sc.drivenByParents) :
  sc.train - sc.drivenByParents = 8 ∧ 
  sc.localStudents + sc.publicTransport + sc.privateTransport = 160 := by
  sorry


end NUMINAMATH_CALUDE_school_commute_theorem_l1946_194609


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1946_194680

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle is tangent to both semicircles and the diameter -/
def is_tangent (config : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ),
    let d := config.R - config.x
    let h := Real.sqrt (d^2 - config.x^2)
    (config.R + config.r)^2 = d^2 + (config.r + config.x)^2 ∧
    h^2 + config.x^2 = config.R^2 ∧
    h^2 + (config.r + config.x)^2 = (config.R + config.r)^2

theorem inscribed_circle_radius
  (config : SemicircleConfig)
  (h₁ : config.R = 18)
  (h₂ : config.r = 9)
  (h₃ : is_tangent config) :
  config.x = 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1946_194680


namespace NUMINAMATH_CALUDE_pure_imaginary_roots_of_f_l1946_194612

/-- The polynomial function we're analyzing -/
def f (x : ℂ) : ℂ := x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_roots_of_f :
  ∃! (z : ℂ), f z = 0 ∧ isPureImaginary z ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_roots_of_f_l1946_194612


namespace NUMINAMATH_CALUDE_odot_two_four_l1946_194655

def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem odot_two_four : odot 2 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_odot_two_four_l1946_194655


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1946_194645

theorem inequality_solution_set (x : ℝ) : x + 8 < 4*x - 1 ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1946_194645


namespace NUMINAMATH_CALUDE_ribbon_length_for_circular_sign_l1946_194679

/-- Given a circular region with area 616 square inches, using π ≈ 22/7,
    and adding 10% extra to the circumference, prove that the amount of
    ribbon needed (rounded up to the nearest inch) is 97 inches. -/
theorem ribbon_length_for_circular_sign :
  let area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let extra_ribbon : ℝ := 0.1 * circumference
  let total_ribbon : ℝ := circumference + extra_ribbon
  ⌈total_ribbon⌉ = 97 := by
sorry

end NUMINAMATH_CALUDE_ribbon_length_for_circular_sign_l1946_194679


namespace NUMINAMATH_CALUDE_walts_interest_l1946_194610

/-- Calculates the total interest earned from two investments with different rates -/
def total_interest (total_amount : ℝ) (amount_at_lower_rate : ℝ) (lower_rate : ℝ) (higher_rate : ℝ) : ℝ :=
  (amount_at_lower_rate * lower_rate) + ((total_amount - amount_at_lower_rate) * higher_rate)

/-- Proves that Walt's total interest is $770 given the problem conditions -/
theorem walts_interest :
  let total_amount : ℝ := 9000
  let amount_at_lower_rate : ℝ := 4000
  let lower_rate : ℝ := 0.08
  let higher_rate : ℝ := 0.09
  total_interest total_amount amount_at_lower_rate lower_rate higher_rate = 770 := by
  sorry

#eval total_interest 9000 4000 0.08 0.09

end NUMINAMATH_CALUDE_walts_interest_l1946_194610


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1946_194684

theorem max_value_of_expression (t : ℝ) : 
  (∃ (t_max : ℝ), ∀ (t : ℝ), ((3^t - 5*t)*t)/(9^t) ≤ ((3^t_max - 5*t_max)*t_max)/(9^t_max)) ∧ 
  (∃ (t_0 : ℝ), ((3^t_0 - 5*t_0)*t_0)/(9^t_0) = 1/20) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1946_194684


namespace NUMINAMATH_CALUDE_no_solution_iff_b_geq_neg_four_thirds_l1946_194672

theorem no_solution_iff_b_geq_neg_four_thirds (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔ 
  b ≥ -4/3 :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_b_geq_neg_four_thirds_l1946_194672


namespace NUMINAMATH_CALUDE_exactly_two_valid_positions_l1946_194660

/-- Represents a position where an additional square can be placed -/
inductive Position
| Left : Position
| Right : Position
| Top : Position
| Bottom : Position
| FrontLeft : Position
| FrontRight : Position

/-- Represents the 'F' shape configuration -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents the modified shape with an additional square -/
structure ModifiedShape :=
  (base : FShape)
  (additional_square : Position)

/-- Predicate to check if a modified shape can be folded into a valid 3D structure -/
def can_fold_to_valid_structure (shape : ModifiedShape) : Prop :=
  sorry

/-- The main theorem stating there are exactly two valid positions -/
theorem exactly_two_valid_positions :
  ∃ (p₁ p₂ : Position), p₁ ≠ p₂ ∧
    (∀ (shape : ModifiedShape),
      can_fold_to_valid_structure shape ↔ shape.additional_square = p₁ ∨ shape.additional_square = p₂) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_positions_l1946_194660


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1946_194616

theorem polynomial_simplification (x : ℝ) :
  5 - 3*x - 7*x^2 + 11 - 5*x + 9*x^2 - 13 + 7*x - 4*x^3 + 7*x^2 + 2*x^3 =
  3 - x + 9*x^2 - 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1946_194616


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l1946_194624

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem one_fifth_of_one_third_of_one_sixth_of_ninety :
  (1 / 5 : ℚ) * ((1 / 3 : ℚ) * ((1 / 6 : ℚ) * 90)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l1946_194624


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l1946_194681

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) ↔ m ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l1946_194681


namespace NUMINAMATH_CALUDE_power_of_product_l1946_194608

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1946_194608


namespace NUMINAMATH_CALUDE_no_zeros_in_2_16_l1946_194626

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having a unique zero point in (0, 2)
def has_unique_zero_in_0_2 (f : ℝ → ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f x = 0

-- Theorem statement
theorem no_zeros_in_2_16 (h : has_unique_zero_in_0_2 f) :
  ∀ x ∈ Set.Ico 2 16, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_in_2_16_l1946_194626


namespace NUMINAMATH_CALUDE_range_of_a_l1946_194619

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, a ≥ -x^2 + 2*x - 2/3) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) ↔ 
  a ∈ Set.Icc (1/3) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1946_194619
