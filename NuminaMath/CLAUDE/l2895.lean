import Mathlib

namespace NUMINAMATH_CALUDE_bank_balance_after_five_years_l2895_289560

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents the bank account balance after each year -/
def bankBalance : ℕ → ℝ
  | 0 => 5600
  | 1 => compoundInterest 5600 0.03 1
  | 2 => compoundInterest (bankBalance 1) 0.035 1
  | 3 => compoundInterest (bankBalance 2 + 2000) 0.04 1
  | 4 => compoundInterest (bankBalance 3) 0.045 1
  | 5 => compoundInterest (bankBalance 4) 0.05 1
  | _ => 0  -- For years beyond 5, return 0

theorem bank_balance_after_five_years :
  bankBalance 5 = 9094.20 := by
  sorry


end NUMINAMATH_CALUDE_bank_balance_after_five_years_l2895_289560


namespace NUMINAMATH_CALUDE_shape_relations_l2895_289573

/-- Given symbols representing geometric shapes with the following relations:
    - triangle + triangle = star
    - circle = square + square
    - triangle = circle + circle + circle + circle
    Prove that star divided by square equals 16 -/
theorem shape_relations (triangle star circle square : ℕ) 
    (h1 : triangle + triangle = star)
    (h2 : circle = square + square)
    (h3 : triangle = circle + circle + circle + circle) :
  star / square = 16 := by sorry

end NUMINAMATH_CALUDE_shape_relations_l2895_289573


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2895_289598

theorem ten_thousandths_digit_of_seven_thirty_seconds (x : ℚ) : 
  x = 7 / 32 → (x * 10000).floor % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2895_289598


namespace NUMINAMATH_CALUDE_triangle_area_ratio_specific_triangle_area_ratio_l2895_289525

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) :
  base > 0 → height1 > 0 → height2 > 0 →
  (base * height1 / 2) / (base * height2 / 2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem specific_triangle_area_ratio :
  let base := 3
  let height1 := 6.02
  let height2 := 2
  (base * height1 / 2) / (base * height2 / 2) = 3.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_specific_triangle_area_ratio_l2895_289525


namespace NUMINAMATH_CALUDE_max_b_value_l2895_289548

/-- The volume of the box -/
def box_volume : ℕ := 360

/-- Theorem stating the maximum possible value of b given the conditions -/
theorem max_b_value (a b c : ℕ) 
  (vol_eq : a * b * c = box_volume)
  (int_cond : 1 < c ∧ c < b ∧ b < a) : 
  b ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l2895_289548


namespace NUMINAMATH_CALUDE_problem_statements_l2895_289507

theorem problem_statements :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, (abs x > abs y ∧ x ≤ y) ∧ ∃ x y : ℝ, (x > y ∧ abs x ≤ abs y)) ∧
  (∃ x : ℤ, x^2 ≤ 0) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x^2 - 2*x + m = 0 ∧ x > 0 ∧ y < 0) ↔ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2895_289507


namespace NUMINAMATH_CALUDE_sin_negative_690_degrees_l2895_289583

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_690_degrees_l2895_289583


namespace NUMINAMATH_CALUDE_ascending_order_real_numbers_l2895_289536

theorem ascending_order_real_numbers : -6 < (0 : ℝ) ∧ 0 < Real.sqrt 5 ∧ Real.sqrt 5 < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_real_numbers_l2895_289536


namespace NUMINAMATH_CALUDE_square_difference_l2895_289549

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2895_289549


namespace NUMINAMATH_CALUDE_apollo_chariot_cost_l2895_289538

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total cost of chariot wheels for Apollo in golden apples for a year -/
def total_cost : ℕ := 
  (months_before_increase * initial_price) + 
  ((months_in_year - months_before_increase) * (initial_price * price_increase_factor))

/-- Theorem stating that the total cost for Apollo is 54 golden apples -/
theorem apollo_chariot_cost : total_cost = 54 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_cost_l2895_289538


namespace NUMINAMATH_CALUDE_stating_head_start_for_tie_l2895_289580

/-- Represents the race scenario -/
structure RaceScenario where
  course_length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- 
Calculates whether the race ends in a tie given a RaceScenario
-/
def is_tie (scenario : RaceScenario) : Prop :=
  scenario.course_length / scenario.speed_ratio = 
  (scenario.course_length - scenario.head_start)

/-- 
Theorem stating that for a 84-meter course where A is 4 times faster than B,
a 63-meter head start results in a tie
-/
theorem head_start_for_tie : 
  let scenario : RaceScenario := {
    course_length := 84,
    speed_ratio := 4,
    head_start := 63
  }
  is_tie scenario := by sorry

end NUMINAMATH_CALUDE_stating_head_start_for_tie_l2895_289580


namespace NUMINAMATH_CALUDE_alpha_third_range_l2895_289540

open Real Set

theorem alpha_third_range (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0) (h3 : sin (α/3) > cos (α/3)) :
  ∃ k : ℤ, α/3 ∈ (Set.Ioo (2*k*π + π/4) (2*k*π + π/3)) ∪ (Set.Ioo (2*k*π + 5*π/6) (2*k*π + π)) :=
sorry

end NUMINAMATH_CALUDE_alpha_third_range_l2895_289540


namespace NUMINAMATH_CALUDE_students_playing_sport_b_l2895_289545

/-- Given that there are 6 students playing sport A, and the number of students
    playing sport B is 4 times the number of students playing sport A,
    prove that 24 students play sport B. -/
theorem students_playing_sport_b (students_a : ℕ) (students_b : ℕ) : 
  students_a = 6 →
  students_b = 4 * students_a →
  students_b = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sport_b_l2895_289545


namespace NUMINAMATH_CALUDE_residue_mod_32_l2895_289578

theorem residue_mod_32 : Int.mod (-1277) 32 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_32_l2895_289578


namespace NUMINAMATH_CALUDE_total_wheels_is_47_l2895_289574

/-- The total number of wheels in Jordan's neighborhood -/
def total_wheels : ℕ :=
  let jordans_driveway := 
    2 * 4 + -- Two cars with 4 wheels each
    1 +     -- One car has a spare wheel
    3 * 2 + -- Three bikes with 2 wheels each
    1 +     -- One bike missing a rear wheel
    3 +     -- One bike with 2 main wheels and one training wheel
    2 +     -- Trash can with 2 wheels
    3 +     -- Tricycle with 3 wheels
    4 +     -- Wheelchair with 2 main wheels and 2 small front wheels
    4 +     -- Wagon with 4 wheels
    3       -- Pair of old roller skates with 3 wheels (one missing)
  let neighbors_driveway :=
    4 +     -- Pickup truck with 4 wheels
    2 +     -- Boat trailer with 2 wheels
    2 +     -- Motorcycle with 2 wheels
    4       -- ATV with 4 wheels
  jordans_driveway + neighbors_driveway

theorem total_wheels_is_47 : total_wheels = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_47_l2895_289574


namespace NUMINAMATH_CALUDE_scrunchies_to_barrettes_ratio_l2895_289514

/-- Represents the number of hair decorations Annie has --/
structure HairDecorations where
  barrettes : ℕ
  scrunchies : ℕ
  bobby_pins : ℕ

/-- Calculates the percentage of bobby pins in the total hair decorations --/
def bobby_pin_percentage (hd : HairDecorations) : ℚ :=
  (hd.bobby_pins : ℚ) / ((hd.barrettes + hd.scrunchies + hd.bobby_pins) : ℚ) * 100

/-- Theorem stating the ratio of scrunchies to barrettes --/
theorem scrunchies_to_barrettes_ratio (hd : HairDecorations) :
  hd.barrettes = 6 →
  hd.bobby_pins = hd.barrettes - 3 →
  bobby_pin_percentage hd = 14 →
  (hd.scrunchies : ℚ) / (hd.barrettes : ℚ) = 2 := by
  sorry

#check scrunchies_to_barrettes_ratio

end NUMINAMATH_CALUDE_scrunchies_to_barrettes_ratio_l2895_289514


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2895_289522

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c,
    then the first component of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) :
  a.1 = Real.sqrt 3 →
  a.2 = 1 →
  b.1 = 0 →
  b.2 = 1 →
  c.2 = Real.sqrt 3 →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  c.1 = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2895_289522


namespace NUMINAMATH_CALUDE_sequence_formula_l2895_289585

/-- Given a sequence {a_n} where the sum of the first n terms S_n = 2^n - 1,
    prove that the general formula for the sequence is a_n = 2^(n-1) -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2^n - 1) : 
    ∀ n : ℕ, a n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2895_289585


namespace NUMINAMATH_CALUDE_vowel_initial_probability_is_7_26_l2895_289512

/-- The set of all letters in the alphabet -/
def Alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

/-- The set of vowels, including Y and W -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y', 'W'}

/-- A student's initials, represented by a single character -/
structure Initial where
  letter : Char
  letter_in_alphabet : letter ∈ Alphabet

/-- The class of students -/
def ClassInitials : Finset Initial := sorry

/-- The number of students in the class -/
axiom class_size : ClassInitials.card = 26

/-- All initials in the class are unique -/
axiom initials_unique : ∀ i j : Initial, i ∈ ClassInitials → j ∈ ClassInitials → i = j → i.letter = j.letter

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability : ℚ :=
  (ClassInitials.filter (fun i => i.letter ∈ Vowels)).card / ClassInitials.card

/-- The main theorem: probability of selecting a student with vowel initials is 7/26 -/
theorem vowel_initial_probability_is_7_26 : vowel_initial_probability = 7 / 26 := by
  sorry

end NUMINAMATH_CALUDE_vowel_initial_probability_is_7_26_l2895_289512


namespace NUMINAMATH_CALUDE_cone_base_radius_l2895_289510

/-- Given a semicircle with radius 6 cm forming the lateral surface of a cone,
    prove that the radius of the base circle of the cone is 3 cm. -/
theorem cone_base_radius (r : ℝ) (h : r = 6) : 
  2 * π * r / 2 = 2 * π * 3 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2895_289510


namespace NUMINAMATH_CALUDE_marble_jar_problem_l2895_289501

/-- The number of marbles in the jar -/
def M : ℕ := 364

/-- The initial number of people -/
def initial_people : ℕ := 26

/-- The number of people who join later -/
def joining_people : ℕ := 2

/-- The number of marbles each person gets in the initial distribution -/
def initial_distribution : ℕ := M / initial_people

/-- The number of marbles each person would get after more people join -/
def later_distribution : ℕ := M / (initial_people + joining_people)

theorem marble_jar_problem :
  (M = initial_people * initial_distribution) ∧
  (M = (initial_people + joining_people) * (initial_distribution - 1)) :=
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l2895_289501


namespace NUMINAMATH_CALUDE_consecutive_primes_integral_roots_properties_l2895_289579

-- Define consecutive primes
def consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

-- Define the quadratic equation with integral roots
def has_integral_roots (p q : ℕ) : Prop :=
  ∃ x y : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 ∧
             y^2 - (p + q : ℤ) * y + (p * q : ℤ) = 0 ∧
             x ≠ y

theorem consecutive_primes_integral_roots_properties
  (p q : ℕ) (h1 : consecutive_primes p q) (h2 : has_integral_roots p q) :
  (∃ x y : ℤ, x + y = p + q ∧ Even (x + y)) ∧  -- Sum of roots is even
  (∀ x : ℤ, x^2 - (p + q : ℤ) * x + (p * q : ℤ) = 0 → x ≥ p) ∧  -- Each root ≥ p
  ¬Nat.Prime (p + q) :=  -- p+q is composite
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_integral_roots_properties_l2895_289579


namespace NUMINAMATH_CALUDE_max_value_product_l2895_289593

theorem max_value_product (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2895_289593


namespace NUMINAMATH_CALUDE_root_of_equation_l2895_289551

def combination (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def permutation (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem root_of_equation : ∃ (x : ℕ), 
  x > 6 ∧ 3 * (combination (x - 3) 4) = 5 * (permutation (x - 4) 2) ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_root_of_equation_l2895_289551


namespace NUMINAMATH_CALUDE_dog_tail_length_l2895_289546

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  overall : ℝ
  body : ℝ
  head : ℝ
  tail : ℝ

/-- Theorem stating the length of a dog's tail given specific proportions -/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.overall = 30)
  (h2 : d.tail = d.body / 2)
  (h3 : d.head = d.body / 6)
  (h4 : d.overall = d.head + d.body + d.tail) : 
  d.tail = 6 := by
  sorry

#check dog_tail_length

end NUMINAMATH_CALUDE_dog_tail_length_l2895_289546


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2895_289529

/-- Given a triangle ABC with side lengths proportional to 7:5:3 and area 45√3,
    prove that the radius of its circumscribed circle is 14. -/
theorem circumradius_of_special_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * a * b * Real.sin C = 45 * Real.sqrt 3 →
  R = (a / (2 * Real.sin A)) →
  R = 14 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2895_289529


namespace NUMINAMATH_CALUDE_wine_barrels_l2895_289547

theorem wine_barrels (a b : ℝ) : 
  (a + 8 = b) ∧ (b + 3 = 3 * (a - 3)) → a = 10 ∧ b = 18 := by
  sorry

end NUMINAMATH_CALUDE_wine_barrels_l2895_289547


namespace NUMINAMATH_CALUDE_sandcastle_problem_l2895_289588

theorem sandcastle_problem (mark_castles : ℕ) : 
  (mark_castles * 10 + mark_castles) +  -- Mark's castles and towers
  ((3 * mark_castles) * 5 + (3 * mark_castles)) = 580 -- Jeff's castles and towers
  → mark_castles = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_problem_l2895_289588


namespace NUMINAMATH_CALUDE_tangent_line_and_critical_point_l2895_289597

/-- The function f(x) = (1/2)x^2 - ax - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x - Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x - a - 1/x

theorem tangent_line_and_critical_point (a : ℝ) (h : a ≥ 0) :
  /- The equation of the tangent line to f(x) at x=1 when a=1 is y = -x + 1/2 -/
  (let y : ℝ → ℝ := fun x ↦ -x + 1/2
   f 1 1 = y 1 ∧ f' 1 1 = -1) ∧
  /- For any critical point x₀ of f(x), f(x₀) ≤ 1/2 -/
  ∀ x₀ > 0, f' a x₀ = 0 → f a x₀ ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_critical_point_l2895_289597


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2895_289559

theorem trig_identity_proof (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * (Real.sin α)^2 - Real.sin α * Real.cos α + (Real.cos α)^2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2895_289559


namespace NUMINAMATH_CALUDE_even_function_iff_a_eq_one_l2895_289599

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_iff_a_eq_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_iff_a_eq_one_l2895_289599


namespace NUMINAMATH_CALUDE_parallelogram_with_inscribed_circle_is_rhombus_l2895_289513

/-- A parallelogram is a quadrilateral with opposite sides parallel and equal. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- A circle is inscribed in a quadrilateral if it touches all four sides. -/
def has_inscribed_circle (p : Parallelogram) : Prop := sorry

/-- A rhombus is a parallelogram with all sides equal. -/
def is_rhombus (p : Parallelogram) : Prop := sorry

/-- Theorem: If a circle can be inscribed in a parallelogram, then the parallelogram is a rhombus. -/
theorem parallelogram_with_inscribed_circle_is_rhombus (p : Parallelogram) :
  has_inscribed_circle p → is_rhombus p := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_with_inscribed_circle_is_rhombus_l2895_289513


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2895_289535

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2895_289535


namespace NUMINAMATH_CALUDE_dog_max_distance_dog_max_distance_is_22_l2895_289518

/-- The maximum distance a dog can reach from the origin when secured at (6,8) with a 12-foot rope -/
theorem dog_max_distance : ℝ :=
  let dog_position : ℝ × ℝ := (6, 8)
  let rope_length : ℝ := 12
  let origin : ℝ × ℝ := (0, 0)
  let distance_to_origin : ℝ := Real.sqrt ((dog_position.1 - origin.1)^2 + (dog_position.2 - origin.2)^2)
  distance_to_origin + rope_length

theorem dog_max_distance_is_22 : dog_max_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_dog_max_distance_dog_max_distance_is_22_l2895_289518


namespace NUMINAMATH_CALUDE_power_of_power_l2895_289595

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2895_289595


namespace NUMINAMATH_CALUDE_louise_wallet_amount_l2895_289575

/-- The amount of money in Louise's wallet --/
def wallet_amount : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_toys, toy_price, num_bears, bear_price =>
    num_toys * toy_price + num_bears * bear_price

/-- Theorem stating the amount in Louise's wallet --/
theorem louise_wallet_amount :
  wallet_amount 28 10 20 15 = 580 := by
  sorry

end NUMINAMATH_CALUDE_louise_wallet_amount_l2895_289575


namespace NUMINAMATH_CALUDE_equation_solution_l2895_289500

theorem equation_solution : ∃ x : ℚ, 
  (((5 - 4*x) / (5 + 4*x) + 3) / (3 + (5 + 4*x) / (5 - 4*x))) - 
  (((5 - 4*x) / (5 + 4*x) + 2) / (2 + (5 + 4*x) / (5 - 4*x))) = 
  ((5 - 4*x) / (5 + 4*x) + 1) / (1 + (5 + 4*x) / (5 - 4*x)) ∧ 
  x = -5/14 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2895_289500


namespace NUMINAMATH_CALUDE_smallest_X_value_l2895_289570

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of X -/
theorem smallest_X_value (T : ℕ) (hT : T > 0) (hComposed : isComposedOf0sAnd1s T) 
    (hDivisible : T % 15 = 0) : 
  ∀ X : ℕ, (X * 15 = T) → X ≥ 7400 := by
  sorry

end NUMINAMATH_CALUDE_smallest_X_value_l2895_289570


namespace NUMINAMATH_CALUDE_equation_equality_l2895_289557

theorem equation_equality : (3 * 6 * 9) / 3 = (2 * 6 * 9) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2895_289557


namespace NUMINAMATH_CALUDE_max_second_term_is_9_l2895_289541

/-- An arithmetic sequence of three positive integers with sum 27 -/
structure ArithSeq27 where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_27 : a + (a + d) + (a + 2*d) = 27

/-- The second term of an arithmetic sequence -/
def second_term (seq : ArithSeq27) : ℕ := seq.a + seq.d

/-- Theorem: The maximum value of the second term in any ArithSeq27 is 9 -/
theorem max_second_term_is_9 : 
  ∀ seq : ArithSeq27, second_term seq ≤ 9 ∧ ∃ seq : ArithSeq27, second_term seq = 9 := by
  sorry

#check max_second_term_is_9

end NUMINAMATH_CALUDE_max_second_term_is_9_l2895_289541


namespace NUMINAMATH_CALUDE_gcd_factorial_nine_eleven_l2895_289527

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_nine_eleven : 
  Nat.gcd (factorial 9) (factorial 11) = factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_nine_eleven_l2895_289527


namespace NUMINAMATH_CALUDE_equation_proof_l2895_289555

theorem equation_proof (x : ℝ) (h : x = 12) : (17.28 / x) / (3.6 * 0.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2895_289555


namespace NUMINAMATH_CALUDE_units_digit_base_6_product_l2895_289505

theorem units_digit_base_6_product (a b : ℕ) (ha : a = 312) (hb : b = 67) :
  (a * b) % 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_base_6_product_l2895_289505


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l2895_289563

theorem second_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n * (2 * n + 1)) → 
  a 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l2895_289563


namespace NUMINAMATH_CALUDE_bank_a_investment_l2895_289520

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  total_investment : ℝ
  bank_a_rate : ℝ
  bank_b_rate : ℝ
  bank_b_fee : ℝ
  years : ℕ
  final_amount : ℝ

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the correct amount invested in Bank A -/
theorem bank_a_investment (scenario : InvestmentScenario) 
  (h1 : scenario.total_investment = 2000)
  (h2 : scenario.bank_a_rate = 0.04)
  (h3 : scenario.bank_b_rate = 0.06)
  (h4 : scenario.bank_b_fee = 50)
  (h5 : scenario.years = 3)
  (h6 : scenario.final_amount = 2430) :
  ∃ (bank_a_amount : ℝ),
    bank_a_amount = 1625 ∧
    compound_interest bank_a_amount scenario.bank_a_rate scenario.years +
    compound_interest (scenario.total_investment - scenario.bank_b_fee - bank_a_amount) scenario.bank_b_rate scenario.years =
    scenario.final_amount :=
  sorry

end NUMINAMATH_CALUDE_bank_a_investment_l2895_289520


namespace NUMINAMATH_CALUDE_video_game_earnings_l2895_289526

/-- Given the conditions of Mike's video game selling scenario, prove the total earnings. -/
theorem video_game_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : 
  total_games = 16 → non_working_games = 8 → price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l2895_289526


namespace NUMINAMATH_CALUDE_right_triangle_area_l2895_289534

/-- The area of a right triangle with one leg measuring 6 and hypotenuse measuring 10 is 24. -/
theorem right_triangle_area : ∀ (a b c : ℝ), 
  a = 6 →
  c = 10 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2895_289534


namespace NUMINAMATH_CALUDE_goose_eggs_count_l2895_289531

theorem goose_eggs_count (
  total_eggs : ℕ
  ) (
  hatched_ratio : Rat
  ) (
  first_month_survival_ratio : Rat
  ) (
  first_year_death_ratio : Rat
  ) (
  first_year_survivors : ℕ
  ) : total_eggs = 2200 :=
  by
  have h1 : hatched_ratio = 2 / 3 := by sorry
  have h2 : first_month_survival_ratio = 3 / 4 := by sorry
  have h3 : first_year_death_ratio = 3 / 5 := by sorry
  have h4 : first_year_survivors = 110 := by sorry
  have h5 : ∀ e, e ≤ 1 := by sorry  -- No more than one goose hatched from each egg
  
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l2895_289531


namespace NUMINAMATH_CALUDE_base7_subtraction_l2895_289532

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The statement to be proved -/
theorem base7_subtraction :
  let a := base7ToDecimal [2, 5, 3, 4]
  let b := base7ToDecimal [1, 4, 6, 6]
  decimalToBase7 (a - b) = [1, 0, 6, 5] := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l2895_289532


namespace NUMINAMATH_CALUDE_hyperbola_I_equation_hyperbola_II_equation_l2895_289594

-- Part I
def hyperbola_I (x y : ℝ) : Prop :=
  let c : ℝ := 8  -- half of focal distance
  let e : ℝ := 4/3  -- eccentricity
  let a : ℝ := c/e
  let b : ℝ := Real.sqrt (c^2 - a^2)
  y^2/a^2 - x^2/b^2 = 1

theorem hyperbola_I_equation : 
  ∀ x y : ℝ, hyperbola_I x y ↔ y^2/36 - x^2/28 = 1 :=
sorry

-- Part II
def hyperbola_II (x y : ℝ) : Prop :=
  let c : ℝ := 6  -- distance from center to focus
  let a : ℝ := Real.sqrt (c^2/2)
  x^2/a^2 - y^2/a^2 = 1

theorem hyperbola_II_equation :
  ∀ x y : ℝ, hyperbola_II x y ↔ x^2/18 - y^2/18 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_I_equation_hyperbola_II_equation_l2895_289594


namespace NUMINAMATH_CALUDE_integer_root_of_special_polynomial_l2895_289533

/-- Given a polynomial with integer coefficients of the form
    x^4 + b_3*x^3 + b_2*x^2 + b_1*x + 50,
    if s is an integer root of this polynomial and s^3 divides 50,
    then s = 1 or s = -1 -/
theorem integer_root_of_special_polynomial (b₃ b₂ b₁ s : ℤ) :
  (s^4 + b₃*s^3 + b₂*s^2 + b₁*s + 50 = 0) →
  (s^3 ∣ 50) →
  (s = 1 ∨ s = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_special_polynomial_l2895_289533


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l2895_289554

/-- Given two circles in a 2D plane, one centered at (1,1) with radius 5 
and another centered at (1,-8) with radius √26, this theorem states that 
the square of the distance between their intersection points is 3128/81. -/
theorem intersection_distance_squared : 
  ∃ (C D : ℝ × ℝ), 
    ((C.1 - 1)^2 + (C.2 - 1)^2 = 25) ∧ 
    ((D.1 - 1)^2 + (D.2 - 1)^2 = 25) ∧
    ((C.1 - 1)^2 + (C.2 + 8)^2 = 26) ∧ 
    ((D.1 - 1)^2 + (D.2 + 8)^2 = 26) ∧
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 3128 / 81) :=
by sorry


end NUMINAMATH_CALUDE_intersection_distance_squared_l2895_289554


namespace NUMINAMATH_CALUDE_intersection_A_B_l2895_289582

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {-1, 1, 2, 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2895_289582


namespace NUMINAMATH_CALUDE_print_statement_output_l2895_289564

def print_output (a : ℕ) : String := s!"a={a}"

theorem print_statement_output (a : ℕ) (h : a = 10) : print_output a = "a=10" := by
  sorry

end NUMINAMATH_CALUDE_print_statement_output_l2895_289564


namespace NUMINAMATH_CALUDE_parcel_delivery_growth_l2895_289571

/-- Represents the equation for parcel delivery growth over three months -/
theorem parcel_delivery_growth 
  (initial_delivery : ℕ) 
  (total_delivery : ℕ) 
  (growth_rate : ℝ) : 
  initial_delivery = 20000 → 
  total_delivery = 72800 → 
  2 + 2 * (1 + growth_rate) + 2 * (1 + growth_rate)^2 = 7.28 := by
  sorry

#check parcel_delivery_growth

end NUMINAMATH_CALUDE_parcel_delivery_growth_l2895_289571


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2895_289568

theorem r_value_when_n_is_3 : 
  let n : ℕ := 3
  let s := 2^n + 2
  let r := 4^s - 2*s
  r = 1048556 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2895_289568


namespace NUMINAMATH_CALUDE_square_root_special_form_l2895_289572

theorem square_root_special_form :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (∃ a b : ℕ, n = 10 * a + b ∧ Real.sqrt n = a + Real.sqrt b) ↔
    (n = 64 ∨ n = 81) := by
  sorry

end NUMINAMATH_CALUDE_square_root_special_form_l2895_289572


namespace NUMINAMATH_CALUDE_book_cost_solution_l2895_289542

def book_cost_problem (x : ℕ) : Prop :=
  x > 0 ∧ 10 * x ≤ 1100 ∧ 11 * x > 1200

theorem book_cost_solution : ∃ (x : ℕ), book_cost_problem x ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_solution_l2895_289542


namespace NUMINAMATH_CALUDE_savings_after_twelve_months_l2895_289524

/-- Represents the electricity pricing and consumption data for a user. -/
structure ElectricityData where
  originalPrice : ℚ
  valleyPrice : ℚ
  peakPrice : ℚ
  installationFee : ℚ
  monthlyConsumption : ℚ
  valleyConsumption : ℚ
  peakConsumption : ℚ
  months : ℕ

/-- Calculates the total savings after a given number of months for a user
    who has installed a peak-valley meter. -/
def totalSavings (data : ElectricityData) : ℚ :=
  let monthlyOriginalCost := data.monthlyConsumption * data.originalPrice
  let monthlyNewCost := data.valleyConsumption * data.valleyPrice + data.peakConsumption * data.peakPrice
  let monthlySavings := monthlyOriginalCost - monthlyNewCost
  let totalSavingsBeforeFee := monthlySavings * data.months
  totalSavingsBeforeFee - data.installationFee

/-- The main theorem stating that the total savings after 12 months is 236 yuan. -/
theorem savings_after_twelve_months :
  let data : ElectricityData := {
    originalPrice := 56/100,
    valleyPrice := 28/100,
    peakPrice := 56/100,
    installationFee := 100,
    monthlyConsumption := 200,
    valleyConsumption := 100,
    peakConsumption := 100,
    months := 12
  }
  totalSavings data = 236 := by sorry

end NUMINAMATH_CALUDE_savings_after_twelve_months_l2895_289524


namespace NUMINAMATH_CALUDE_participant_selection_count_l2895_289543

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_participants : ℕ := 4

def select_participants (boys girls participants : ℕ) : ℕ :=
  (Nat.choose boys 3 * Nat.choose girls 1) +
  (Nat.choose boys 2 * Nat.choose girls 2) +
  (Nat.choose boys 1 * Nat.choose girls 3)

theorem participant_selection_count :
  select_participants num_boys num_girls num_participants = 34 := by
  sorry

end NUMINAMATH_CALUDE_participant_selection_count_l2895_289543


namespace NUMINAMATH_CALUDE_max_value_f_positive_three_distinct_roots_condition_l2895_289592

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x > 0 then 1 - x^2 * Real.log x else Real.exp (-x - 2)

-- Part 1: Maximum value of f(x) for x > 0
theorem max_value_f_positive (x : ℝ) (h : x > 0) :
  f x ≤ 1 + 1 / (2 * Real.exp 1) :=
sorry

-- Part 2: Condition for three distinct real roots
theorem three_distinct_roots_condition (a b : ℝ) (h : a ≥ 0) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x + a * x^2 + b * x = 0 ∧
    f y + a * y^2 + b * y = 0 ∧
    f z + a * z^2 + b * z = 0) ↔
  b < -2 * Real.sqrt 2 ∨ b ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_positive_three_distinct_roots_condition_l2895_289592


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l2895_289577

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_three_digit_product (n x y : ℕ) :
  (100 ≤ n ∧ n < 1000) →
  (x < 10 ∧ y < 10) →
  isPrime x →
  isPrime (10 * x + y) →
  n = x * (10 * x + y) →
  n ≤ 553 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l2895_289577


namespace NUMINAMATH_CALUDE_sand_art_problem_l2895_289502

/-- The amount of sand needed to fill one square inch -/
def sand_per_square_inch (rectangle_length rectangle_width square_side total_sand : ℕ) : ℚ :=
  total_sand / (rectangle_length * rectangle_width + square_side * square_side)

theorem sand_art_problem (rectangle_length rectangle_width square_side total_sand : ℕ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 7)
  (h3 : square_side = 5)
  (h4 : total_sand = 201) :
  sand_per_square_inch rectangle_length rectangle_width square_side total_sand = 3 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_problem_l2895_289502


namespace NUMINAMATH_CALUDE_smallest_AC_l2895_289515

/-- Represents a right triangle ABC with a point D on AC -/
structure RightTriangleWithPoint where
  AC : ℕ  -- Length of AC
  CD : ℕ  -- Length of CD
  bd_squared : ℕ  -- Square of length BD

/-- Defines the conditions for the right triangle and point D -/
def valid_triangle (t : RightTriangleWithPoint) : Prop :=
  t.AC > 0 ∧ t.CD > 0 ∧ t.CD < t.AC ∧ t.bd_squared = 36 ∧
  2 * t.AC * t.CD = t.CD * t.CD + t.bd_squared

/-- Theorem: The smallest possible value of AC is 6 -/
theorem smallest_AC :
  ∀ t : RightTriangleWithPoint, valid_triangle t → t.AC ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_AC_l2895_289515


namespace NUMINAMATH_CALUDE_find_d_when_a_b_c_equal_l2895_289581

theorem find_d_when_a_b_c_equal (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + 3 * Real.sqrt (a + b + c - d) →
  a = b →
  b = c →
  d = 5/4 := by
sorry

end NUMINAMATH_CALUDE_find_d_when_a_b_c_equal_l2895_289581


namespace NUMINAMATH_CALUDE_pen_cost_problem_l2895_289511

theorem pen_cost_problem (total_students : Nat) (buyers : Nat) (pens_per_student : Nat) (pen_cost : Nat) :
  total_students = 32 →
  buyers > total_students / 2 →
  pens_per_student > 1 →
  pen_cost > pens_per_student →
  buyers * pens_per_student * pen_cost = 2116 →
  pen_cost = 23 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_problem_l2895_289511


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2895_289517

theorem system_of_equations_solution (a b c : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + 2*z = a) ∧ 
    (x + 2*y + z = b) ∧ 
    (2*x + y + z = c) ∧
    (x = (3*c - a - b) / 4) ∧
    (y = (3*b - a - c) / 4) ∧
    (z = (3*a - b - c) / 4) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2895_289517


namespace NUMINAMATH_CALUDE_product_of_fractions_l2895_289556

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2895_289556


namespace NUMINAMATH_CALUDE_odd_factorial_product_equals_sum_factorial_l2895_289537

def oddFactorialProduct (m : ℕ) : ℕ := (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sumFirstNaturals (m : ℕ) : ℕ := m * (m + 1) / 2

theorem odd_factorial_product_equals_sum_factorial (m : ℕ) :
  oddFactorialProduct m = Nat.factorial (sumFirstNaturals m) ↔ m = 1 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_factorial_product_equals_sum_factorial_l2895_289537


namespace NUMINAMATH_CALUDE_cl2_moles_in_reaction_l2895_289590

/-- Represents the stoichiometric coefficients of the reaction CH4 + 2Cl2 → CHCl3 + 4HCl -/
structure ReactionCoefficients where
  ch4 : ℕ
  cl2 : ℕ
  chcl3 : ℕ
  hcl : ℕ

/-- The balanced equation coefficients for the reaction -/
def balancedEquation : ReactionCoefficients :=
  { ch4 := 1, cl2 := 2, chcl3 := 1, hcl := 4 }

/-- Calculates the moles of Cl2 combined given the moles of CH4 and HCl -/
def molesOfCl2Combined (molesCH4 : ℕ) (molesHCl : ℕ) : ℕ :=
  (balancedEquation.cl2 * molesHCl) / balancedEquation.hcl

theorem cl2_moles_in_reaction (molesCH4 : ℕ) (molesHCl : ℕ) :
  molesCH4 = balancedEquation.ch4 ∧ molesHCl = balancedEquation.hcl →
  molesOfCl2Combined molesCH4 molesHCl = balancedEquation.cl2 :=
by
  sorry

end NUMINAMATH_CALUDE_cl2_moles_in_reaction_l2895_289590


namespace NUMINAMATH_CALUDE_expected_value_proof_l2895_289539

/-- The expected value of winning (6-n)^2 dollars when rolling a fair 6-sided die -/
def expected_value : ℚ := 55 / 6

/-- A fair 6-sided die -/
def die : Finset ℕ := Finset.range 6

/-- The probability of rolling any number on a fair 6-sided die -/
def prob (n : ℕ) : ℚ := 1 / 6

/-- The winnings for rolling n on the die -/
def winnings (n : ℕ) : ℚ := (6 - n) ^ 2

theorem expected_value_proof :
  Finset.sum die (λ n => prob n * winnings n) = expected_value :=
sorry

end NUMINAMATH_CALUDE_expected_value_proof_l2895_289539


namespace NUMINAMATH_CALUDE_total_volume_calculation_l2895_289523

-- Define the dimensions of the rectangular parallelepiped
def box_length : ℝ := 2
def box_width : ℝ := 3
def box_height : ℝ := 4

-- Define the radius of half-spheres and cylinders
def sphere_radius : ℝ := 1
def cylinder_radius : ℝ := 1

-- Define the number of vertices and edges
def num_vertices : ℕ := 8
def num_edges : ℕ := 12

-- Theorem statement
theorem total_volume_calculation :
  let box_volume := box_length * box_width * box_height
  let half_sphere_volume := (num_vertices : ℝ) * (1/2) * (4/3) * Real.pi * sphere_radius^3
  let cylinder_volume := Real.pi * cylinder_radius^2 * 
    (2 * box_length + 2 * box_width + 2 * box_height)
  let total_volume := box_volume + half_sphere_volume + cylinder_volume
  total_volume = (72 + 112 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_total_volume_calculation_l2895_289523


namespace NUMINAMATH_CALUDE_julia_grocery_purchase_l2895_289566

/-- Represents the cost of items and the total bill for Julia's grocery purchase. -/
def grocery_bill (snickers_cost : ℚ) : ℚ :=
  let mms_cost := 2 * snickers_cost
  let pepsi_cost := 2 * mms_cost
  let bread_cost := 3 * pepsi_cost
  2 * snickers_cost + 3 * mms_cost + 4 * pepsi_cost + 5 * bread_cost

/-- Theorem stating the total cost of Julia's purchase and the additional amount she needs to pay. -/
theorem julia_grocery_purchase (snickers_cost : ℚ) (h : snickers_cost = 3/2) :
  grocery_bill snickers_cost = 126 ∧ grocery_bill snickers_cost - 100 = 26 := by
  sorry

#eval grocery_bill (3/2)

end NUMINAMATH_CALUDE_julia_grocery_purchase_l2895_289566


namespace NUMINAMATH_CALUDE_circle_condition_l2895_289586

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation :=
  (a b c d e f : ℝ)

/-- Checks if a QuadraticEquation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.d^2 + eq.e^2 - 4*eq.a*eq.f > 0

/-- The equation m^2x^2 + (m+2)y^2 + 2mx + m = 0 -/
def equation (m : ℝ) : QuadraticEquation :=
  ⟨m^2, m+2, 0, 2*m, 0, m⟩

/-- Theorem: The equation represents a circle if and only if m = -1 -/
theorem circle_condition :
  ∀ m : ℝ, isCircle (equation m) ↔ m = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2895_289586


namespace NUMINAMATH_CALUDE_students_history_or_geography_not_both_l2895_289567

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The total number of students taking history -/
def history : ℕ := 30

/-- The number of students taking only geography -/
def only_geography : ℕ := 18

/-- Theorem: The number of students taking history or geography but not both is 33 -/
theorem students_history_or_geography_not_both : 
  (history - both) + only_geography = 33 := by sorry

end NUMINAMATH_CALUDE_students_history_or_geography_not_both_l2895_289567


namespace NUMINAMATH_CALUDE_f_minimum_and_a_range_l2895_289508

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_a_range :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 1 → f x ≥ a * x - 1) ↔ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_and_a_range_l2895_289508


namespace NUMINAMATH_CALUDE_power_of_power_l2895_289528

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2895_289528


namespace NUMINAMATH_CALUDE_age_difference_l2895_289565

/-- Represents a person's age at different points in time -/
structure AgeRelation where
  current : ℕ
  future : ℕ

/-- The age relation between two people A and B -/
def age_relation (a b : AgeRelation) : Prop :=
  a.current - b.current = b.current - 10 ∧
  a.current - b.current = 25 - a.future

theorem age_difference (a b : AgeRelation) 
  (h : age_relation a b) : a.current - b.current = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2895_289565


namespace NUMINAMATH_CALUDE_number_of_towns_l2895_289503

theorem number_of_towns (n : ℕ) : Nat.choose n 2 = 15 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_towns_l2895_289503


namespace NUMINAMATH_CALUDE_ab_equals_six_l2895_289561

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l2895_289561


namespace NUMINAMATH_CALUDE_charles_learning_time_l2895_289509

/-- The number of days it takes to learn one vowel, given the total days and number of vowels -/
def days_per_vowel (total_days : ℕ) (num_vowels : ℕ) : ℕ :=
  total_days / num_vowels

/-- Theorem stating that it takes 7 days to learn one vowel -/
theorem charles_learning_time :
  days_per_vowel 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_charles_learning_time_l2895_289509


namespace NUMINAMATH_CALUDE_line_segment_params_sum_of_squares_l2895_289521

/-- Given two points in 2D space, this function returns the parameters of the line segment connecting them. -/
def lineSegmentParams (p1 p2 : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem line_segment_params_sum_of_squares :
  let p1 : ℝ × ℝ := (-3, 6)
  let p2 : ℝ × ℝ := (4, 14)
  let (a, b, c, d) := lineSegmentParams p1 p2
  a^2 + b^2 + c^2 + d^2 = 158 := by sorry

end NUMINAMATH_CALUDE_line_segment_params_sum_of_squares_l2895_289521


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2895_289558

/-- Given a cylinder with height 5 inches and radius r, 
    if increasing the radius by 4 inches or increasing the height by 4 inches 
    results in the same volume, then r = 5 + 3√5 -/
theorem cylinder_radius_problem (r : ℝ) : 
  (π * (r + 4)^2 * 5 = π * r^2 * 9) → r = 5 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2895_289558


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_m_range_l2895_289596

theorem quadratic_two_zeros_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) →
  m < -2 ∨ m > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_m_range_l2895_289596


namespace NUMINAMATH_CALUDE_wire_service_reporters_theorem_l2895_289504

/-- Represents the percentage of reporters in a wire service -/
structure ReporterPercentage where
  local_politics : Real
  not_politics : Real
  politics_not_local : Real

/-- Given the percentages of reporters covering local politics and not covering politics,
    calculates the percentage of reporters covering politics but not local politics -/
def calculate_politics_not_local (rp : ReporterPercentage) : Real :=
  100 - rp.not_politics - rp.local_politics

/-- Theorem stating that given the specific percentages in the problem,
    the percentage of reporters covering politics but not local politics is 2.14285714285714% -/
theorem wire_service_reporters_theorem (rp : ReporterPercentage)
  (h1 : rp.local_politics = 5)
  (h2 : rp.not_politics = 92.85714285714286) :
  calculate_politics_not_local rp = 2.14285714285714 := by
  sorry

#eval calculate_politics_not_local { local_politics := 5, not_politics := 92.85714285714286, politics_not_local := 0 }

end NUMINAMATH_CALUDE_wire_service_reporters_theorem_l2895_289504


namespace NUMINAMATH_CALUDE_prob_three_ones_in_four_rolls_eq_5_324_l2895_289506

/-- A fair, regular six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a fair die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of rolling a 1 exactly three times in four rolls of a fair die -/
def prob_three_ones_in_four_rolls : ℚ :=
  (choose 4 3 : ℚ) * (prob {0})^3 * (1 - prob {0})

theorem prob_three_ones_in_four_rolls_eq_5_324 :
  prob_three_ones_in_four_rolls = 5 / 324 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_ones_in_four_rolls_eq_5_324_l2895_289506


namespace NUMINAMATH_CALUDE_average_score_is_106_l2895_289553

/-- The average bowling score of three bowlers -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of three bowlers with scores 120, 113, and 85 is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_106_l2895_289553


namespace NUMINAMATH_CALUDE_f_difference_l2895_289530

/-- The function f defined as f(x) = 5x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(10x + 5h - 2) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2895_289530


namespace NUMINAMATH_CALUDE_value_of_a_l2895_289591

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given value in paise
def given_paise : ℚ := 85

-- Theorem statement
theorem value_of_a (a : ℚ) : given_percentage * a = given_paise → a = 170 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2895_289591


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2895_289576

theorem smallest_number_of_eggs (total_eggs : ℕ) (num_containers : ℕ) : 
  total_eggs > 150 →
  total_eggs = 12 * num_containers - 3 →
  (∀ n : ℕ, n < num_containers → 12 * n - 3 ≤ 150) →
  total_eggs = 153 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2895_289576


namespace NUMINAMATH_CALUDE_expand_product_l2895_289550

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2895_289550


namespace NUMINAMATH_CALUDE_linear_function_composition_l2895_289562

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 3) →
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) :=
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2895_289562


namespace NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l2895_289587

theorem roots_of_x_squared_equals_x :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l2895_289587


namespace NUMINAMATH_CALUDE_expression_simplification_l2895_289589

theorem expression_simplification (x y : ℝ) (h : x^2 ≠ y^2) :
  ((x^2 + y^2) / (x^2 - y^2)) + ((x^2 - y^2) / (x^2 + y^2)) = 2*(x^4 + y^4) / (x^4 - y^4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2895_289589


namespace NUMINAMATH_CALUDE_max_sales_price_l2895_289519

/-- Represents the sales function for a product -/
def sales_function (x : ℝ) : ℝ := 400 - 20 * (x - 30)

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (sales_function x)

/-- The unit purchase price of the product -/
def purchase_price : ℝ := 20

/-- The initial selling price of the product -/
def initial_price : ℝ := 30

/-- The initial sales volume in half a month -/
def initial_volume : ℝ := 400

/-- The price-volume relationship: change in volume per unit price increase -/
def price_volume_ratio : ℝ := -20

theorem max_sales_price : 
  ∃ (x : ℝ), x = 35 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x :=
by sorry

end NUMINAMATH_CALUDE_max_sales_price_l2895_289519


namespace NUMINAMATH_CALUDE_vector_perpendicular_l2895_289544

/-- Given plane vectors a, b, and c, where c is perpendicular to (a + b), prove that t = -6/5 -/
theorem vector_perpendicular (a b c : ℝ × ℝ) (t : ℝ) :
  a = (1, 2) →
  b = (3, 4) →
  c = (t, t + 2) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  t = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l2895_289544


namespace NUMINAMATH_CALUDE_max_a_bound_l2895_289552

theorem max_a_bound (a : ℝ) : 
  (∀ x > 0, (x^2 + 1) * Real.exp x ≥ a * x^2) ↔ a ≤ 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_max_a_bound_l2895_289552


namespace NUMINAMATH_CALUDE_marys_weight_l2895_289569

-- Define the weights as real numbers
variable (mary_weight : ℝ)
variable (john_weight : ℝ)
variable (jamison_weight : ℝ)

-- Define the conditions
axiom john_weight_relation : john_weight = mary_weight + (1/4 * mary_weight)
axiom mary_jamison_relation : mary_weight = jamison_weight - 20
axiom total_weight : mary_weight + john_weight + jamison_weight = 540

-- Theorem to prove
theorem marys_weight : mary_weight = 160 := by
  sorry

end NUMINAMATH_CALUDE_marys_weight_l2895_289569


namespace NUMINAMATH_CALUDE_delivery_fee_calculation_delivery_fee_is_twenty_l2895_289516

theorem delivery_fee_calculation (sandwich_price : ℝ) (num_sandwiches : ℕ) 
  (tip_percentage : ℝ) (total_received : ℝ) (delivery_fee : ℝ) : Prop :=
  sandwich_price = 5 →
  num_sandwiches = 18 →
  tip_percentage = 0.1 →
  total_received = 121 →
  delivery_fee = 20 →
  total_received = (sandwich_price * num_sandwiches) + delivery_fee + 
    (tip_percentage * (sandwich_price * num_sandwiches + delivery_fee))

-- Proof
theorem delivery_fee_is_twenty :
  ∃ (delivery_fee : ℝ),
    delivery_fee_calculation 5 18 0.1 121 delivery_fee :=
by
  sorry

end NUMINAMATH_CALUDE_delivery_fee_calculation_delivery_fee_is_twenty_l2895_289516


namespace NUMINAMATH_CALUDE_square_side_length_l2895_289584

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 81) (h2 : area = side ^ 2) :
  side = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2895_289584
