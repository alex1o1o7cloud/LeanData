import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_eight_to_nine_thirds_l3426_342639

theorem evaluate_eight_to_nine_thirds : 8^(9/3) = 512 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_eight_to_nine_thirds_l3426_342639


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3426_342665

theorem gcd_of_powers_of_two : Nat.gcd (2^1015 - 1) (2^1024 - 1) = 2^9 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3426_342665


namespace NUMINAMATH_CALUDE_product_72516_9999_l3426_342620

theorem product_72516_9999 : 72516 * 9999 = 724987484 := by
  sorry

end NUMINAMATH_CALUDE_product_72516_9999_l3426_342620


namespace NUMINAMATH_CALUDE_minus_one_power_difference_l3426_342659

theorem minus_one_power_difference : (-1)^2024 - (-1)^2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_minus_one_power_difference_l3426_342659


namespace NUMINAMATH_CALUDE_solution_pairs_l3426_342648

theorem solution_pairs (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) →
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1)) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l3426_342648


namespace NUMINAMATH_CALUDE_calligraphy_book_characters_l3426_342690

/-- The number of characters written per day in the first practice -/
def first_practice_chars_per_day : ℕ := 25

/-- The additional characters written per day in the second practice -/
def additional_chars_per_day : ℕ := 3

/-- The number of days fewer in the second practice compared to the first -/
def days_difference : ℕ := 3

/-- The total number of characters in the book -/
def total_characters : ℕ := 700

theorem calligraphy_book_characters :
  ∃ (x : ℕ), 
    x > days_difference ∧
    first_practice_chars_per_day * x = 
    (first_practice_chars_per_day + additional_chars_per_day) * (x - days_difference) ∧
    total_characters = first_practice_chars_per_day * x :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_book_characters_l3426_342690


namespace NUMINAMATH_CALUDE_selection_theorem_l3426_342619

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the number of course representatives to be selected -/
def num_representatives : ℕ := 5

/-- Calculates the number of ways to select representatives under condition I -/
def selection_ways_I : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition II -/
def selection_ways_II : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition III -/
def selection_ways_III : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition IV -/
def selection_ways_IV : ℕ := sorry

theorem selection_theorem :
  selection_ways_I = 840 ∧
  selection_ways_II = 3360 ∧
  selection_ways_III = 5400 ∧
  selection_ways_IV = 1080 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3426_342619


namespace NUMINAMATH_CALUDE_smallest_X_value_l3426_342673

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of X -/
theorem smallest_X_value (T : ℕ) (hT : T > 0) (hComposed : isComposedOf0sAnd1s T) 
    (hDivisible : T % 15 = 0) : 
  ∀ X : ℕ, (X * 15 = T) → X ≥ 7400 := by
  sorry

end NUMINAMATH_CALUDE_smallest_X_value_l3426_342673


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l3426_342658

theorem same_color_plate_probability :
  let total_plates : ℕ := 12
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  (red_combinations + blue_combinations : ℚ) / total_combinations = 9 / 44 :=
by sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l3426_342658


namespace NUMINAMATH_CALUDE_sin_cos_sum_10_20_l3426_342696

theorem sin_cos_sum_10_20 : 
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_10_20_l3426_342696


namespace NUMINAMATH_CALUDE_base7_subtraction_l3426_342605

/-- Represents a number in base 7 as a list of digits (least significant first) -/
def Base7 := List Nat

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : Base7) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers -/
def base7Difference (a b : Base7) : Base7 :=
  sorry -- Implementation not required for the statement

/-- Statement: The difference between 4512₇ and 2345₇ in base 7 is 2144₇ -/
theorem base7_subtraction :
  base7Difference [2, 1, 5, 4] [5, 4, 3, 2] = [4, 4, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base7_subtraction_l3426_342605


namespace NUMINAMATH_CALUDE_parabola_shift_l3426_342615

/-- The equation of a parabola after horizontal and vertical shifts -/
def shifted_parabola (a b c : ℝ) (x y : ℝ) : Prop :=
  y = a * (x - b)^2 + c

theorem parabola_shift :
  ∀ (x y : ℝ),
  (y = 3 * x^2) →  -- Original parabola
  (shifted_parabola 3 1 (-2) x y)  -- Shifted parabola
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3426_342615


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3426_342693

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3426_342693


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3426_342618

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 17 →
  a * b + c + d = 85 →
  a * d + b * c = 196 →
  c * d = 120 →
  a^2 + b^2 + c^2 + d^2 ≤ 918 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3426_342618


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircles_l3426_342641

/-- Given a square with side length 4 and four semicircles with centers at the midpoints of the square's sides, 
    prove that the area not covered by the semicircles is 8 - 2π. -/
theorem shaded_area_in_square_with_semicircles (square_side : ℝ) (semicircle_radius : ℝ) : 
  square_side = 4 → 
  semicircle_radius = Real.sqrt 2 →
  (4 : ℝ) * (π / 2 * semicircle_radius^2) = 2 * π →
  square_side^2 - (4 : ℝ) * (π / 2 * semicircle_radius^2) = 8 - 2 * π := by
  sorry

#align shaded_area_in_square_with_semicircles shaded_area_in_square_with_semicircles

end NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircles_l3426_342641


namespace NUMINAMATH_CALUDE_f_symmetry_l3426_342667

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_symmetry (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3426_342667


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3426_342653

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3426_342653


namespace NUMINAMATH_CALUDE_max_odd_group_length_l3426_342687

/-- A sequence of consecutive natural numbers where each number
    is a product of prime factors with odd exponents -/
def OddGroup (start : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    ∀ p : ℕ, Nat.Prime p → 
      Odd ((start + k).factorization p)

/-- The maximum length of an OddGroup -/
theorem max_odd_group_length : 
  (∃ (max : ℕ), ∀ (start n : ℕ), OddGroup start n → n ≤ max) ∧ 
  (∃ (start : ℕ), OddGroup start 7) :=
sorry

end NUMINAMATH_CALUDE_max_odd_group_length_l3426_342687


namespace NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l3426_342681

theorem tan_30_plus_4cos_30 :
  Real.tan (30 * π / 180) + 4 * Real.cos (30 * π / 180) = 7 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l3426_342681


namespace NUMINAMATH_CALUDE_four_numbers_with_consecutive_sums_l3426_342630

theorem four_numbers_with_consecutive_sums : ∃ (a b c d : ℕ),
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) ∧
  (a + b = 2023) ∧
  (a + c = 2024) ∧
  (a + d = 2026) ∧
  (b + c = 2025) ∧
  (b + d = 2027) ∧
  (c + d = 2028) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_consecutive_sums_l3426_342630


namespace NUMINAMATH_CALUDE_distribution_property_l3426_342610

-- Define a type for our distribution
def Distribution (α : Type*) := α → ℝ

-- Define properties of our distribution
def IsSymmetric (f : Distribution ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def StandardDeviationProperty (f : Distribution ℝ) (a d : ℝ) : Prop :=
  ∫ x in Set.Icc (a - d) (a + d), f x = 0.68

-- Main theorem
theorem distribution_property (f : Distribution ℝ) (a d : ℝ) 
  (h_symmetric : IsSymmetric f a) 
  (h_std_dev : StandardDeviationProperty f a d) :
  ∫ x in Set.Iic (a + d), f x = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_distribution_property_l3426_342610


namespace NUMINAMATH_CALUDE_cubical_box_immersion_l3426_342652

/-- The edge length of a cubical box that, when immersed in a rectangular vessel,
    causes a specific rise in water level. -/
def cubical_box_edge_length (vessel_length vessel_width water_rise : ℝ) : ℝ :=
  (vessel_length * vessel_width * water_rise) ^ (1/3)

/-- Theorem stating that a cubical box with edge length 30 cm, when immersed in a
    vessel of 60 cm by 30 cm, causes the water to rise by 15 cm. -/
theorem cubical_box_immersion (vessel_length vessel_width water_rise : ℝ)
  (h1 : vessel_length = 60)
  (h2 : vessel_width = 30)
  (h3 : water_rise = 15) :
  cubical_box_edge_length vessel_length vessel_width water_rise = 30 :=
by
  sorry

#eval cubical_box_edge_length 60 30 15

end NUMINAMATH_CALUDE_cubical_box_immersion_l3426_342652


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3426_342664

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt 2 * Real.sin x

theorem max_min_f_on_interval :
  let a := 0
  let b := Real.pi
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = Real.pi ∧
    f x_min = Real.pi / 4 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3426_342664


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l3426_342650

/-- Calculates the remaining money after Amanda's purchases -/
def remaining_money (gift_amount : ℚ) (tape_price : ℚ) (num_tapes : ℕ) 
  (headphone_price : ℚ) (vinyl_price : ℚ) (poster_price : ℚ) 
  (tape_discount : ℚ) (headphone_tax : ℚ) (shipping_cost : ℚ) : ℚ :=
  let tape_total := tape_price * num_tapes * (1 - tape_discount)
  let headphone_total := headphone_price * (1 + headphone_tax)
  let total_cost := tape_total + headphone_total + vinyl_price + poster_price + shipping_cost
  gift_amount - total_cost

/-- Theorem stating that Amanda will have $16.75 left after her purchases -/
theorem amanda_remaining_money :
  remaining_money 200 15 3 55 35 45 0.1 0.05 5 = 16.75 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l3426_342650


namespace NUMINAMATH_CALUDE_pizza_parlor_cost_theorem_l3426_342677

/-- Calculates the total cost including gratuity for a group celebration at a pizza parlor -/
def pizza_parlor_cost (total_people : ℕ) (child_pizza_cost adult_pizza_cost child_drink_cost adult_drink_cost : ℚ) (gratuity_rate : ℚ) : ℚ :=
  let num_adults : ℕ := total_people / 3
  let num_children : ℕ := 2 * num_adults
  let child_cost : ℚ := num_children * (child_pizza_cost + child_drink_cost)
  let adult_cost : ℚ := num_adults * (adult_pizza_cost + adult_drink_cost)
  let subtotal : ℚ := child_cost + adult_cost
  let gratuity : ℚ := subtotal * gratuity_rate
  subtotal + gratuity

/-- The total cost including gratuity for the group celebration at the pizza parlor is $1932 -/
theorem pizza_parlor_cost_theorem : 
  pizza_parlor_cost 120 10 12 3 4 (15/100) = 1932 :=
by sorry

end NUMINAMATH_CALUDE_pizza_parlor_cost_theorem_l3426_342677


namespace NUMINAMATH_CALUDE_division_remainder_l3426_342627

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3426_342627


namespace NUMINAMATH_CALUDE_pond_water_after_evaporation_l3426_342632

/-- Calculates the remaining water in a pond after evaporation --/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem: The pond contains 205 gallons after 45 days --/
theorem pond_water_after_evaporation :
  remaining_water 250 1 45 = 205 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_after_evaporation_l3426_342632


namespace NUMINAMATH_CALUDE_min_value_fraction_l3426_342642

theorem min_value_fraction (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = 2 → 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 2 → 8 / ((x + 2) * (y + 4)) ≤ 8 / ((a + 2) * (b + 4))) →
  8 / ((x + 2) * (y + 4)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3426_342642


namespace NUMINAMATH_CALUDE_no_squares_end_in_seven_l3426_342666

theorem no_squares_end_in_seven : 
  ∀ n : ℕ, ¬(∃ m : ℕ, m * m = 10 * n + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_no_squares_end_in_seven_l3426_342666


namespace NUMINAMATH_CALUDE_counterexample_exists_l3426_342612

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a / c ∧ a / b ≠ b / c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3426_342612


namespace NUMINAMATH_CALUDE_mollys_age_l3426_342692

theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 38 →
  molly_age = 24 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l3426_342692


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3426_342600

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 / (2 - I) → z.im = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3426_342600


namespace NUMINAMATH_CALUDE_number_relationship_l3426_342602

theorem number_relationship (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha2 : a^2 = 2) (hb3 : b^3 = 3) (hc4 : c^4 = 4) (hd5 : d^5 = 5) :
  a = c ∧ c < d ∧ d < b :=
sorry

end NUMINAMATH_CALUDE_number_relationship_l3426_342602


namespace NUMINAMATH_CALUDE_prism_height_l3426_342661

/-- A triangular prism with given dimensions -/
structure TriangularPrism where
  volume : ℝ
  base_side1 : ℝ
  base_side2 : ℝ
  height : ℝ

/-- The volume of a triangular prism is equal to the area of its base times its height -/
axiom volume_formula (p : TriangularPrism) : 
  p.volume = (1/2) * p.base_side1 * p.base_side2 * p.height

/-- Theorem: Given a triangular prism with volume 120 cm³ and base sides 3 cm and 4 cm, 
    its height is 20 cm -/
theorem prism_height (p : TriangularPrism) 
  (h_volume : p.volume = 120)
  (h_base1 : p.base_side1 = 3)
  (h_base2 : p.base_side2 = 4) :
  p.height = 20 := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l3426_342661


namespace NUMINAMATH_CALUDE_allocation_methods_for_three_schools_l3426_342606

/-- The number of ways to allocate doctors and nurses to schools. -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.factorial) * (num_nurses.choose 2 * (num_nurses - 2).choose 2)

/-- Theorem stating that there are 540 different allocation methods for 3 doctors and 6 nurses to 3 schools. -/
theorem allocation_methods_for_three_schools :
  allocation_methods 3 6 3 = 540 := by
  sorry

#eval allocation_methods 3 6 3

end NUMINAMATH_CALUDE_allocation_methods_for_three_schools_l3426_342606


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3426_342654

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3426_342654


namespace NUMINAMATH_CALUDE_red_balls_count_l3426_342678

/-- Given a bag of balls with the following properties:
  * The total number of balls is 60
  * The frequency of picking red balls is 0.15
  Prove that the number of red balls in the bag is 9 -/
theorem red_balls_count (total_balls : ℕ) (red_frequency : ℝ) 
  (h1 : total_balls = 60)
  (h2 : red_frequency = 0.15) :
  ⌊total_balls * red_frequency⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3426_342678


namespace NUMINAMATH_CALUDE_square_root_equation_l3426_342645

theorem square_root_equation (n : ℝ) : 3 * Real.sqrt (8 + n) = 15 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3426_342645


namespace NUMINAMATH_CALUDE_total_digits_first_2500_even_integers_l3426_342611

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 2500th positive even integer -/
def nthEvenInteger : ℕ := 5000

theorem total_digits_first_2500_even_integers :
  sumDigitsEven nthEvenInteger = 9448 := by sorry

end NUMINAMATH_CALUDE_total_digits_first_2500_even_integers_l3426_342611


namespace NUMINAMATH_CALUDE_product_726_4_base9_l3426_342617

/-- Convert a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Convert a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

/-- Theorem stating that the product of 726₉ and 4₉ is equal to 3216₉ in base 9. -/
theorem product_726_4_base9 :
  base9ToNat [6, 2, 7] * base9ToNat [4] = base9ToNat [6, 1, 2, 3] := by
  sorry

#eval base9ToNat [6, 2, 7] * base9ToNat [4] == base9ToNat [6, 1, 2, 3]

end NUMINAMATH_CALUDE_product_726_4_base9_l3426_342617


namespace NUMINAMATH_CALUDE_distinct_tetrahedra_count_l3426_342697

/-- A type representing a thin rod with a length -/
structure Rod where
  length : ℝ
  positive : length > 0

/-- A type representing a set of 6 rods -/
structure SixRods where
  rods : Fin 6 → Rod
  distinct : ∀ i j, i ≠ j → rods i ≠ rods j

/-- A predicate stating that any three rods can form a triangle -/
def can_form_triangle (sr : SixRods) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (sr.rods i).length + (sr.rods j).length > (sr.rods k).length

/-- A type representing a tetrahedral edge framework -/
structure Tetrahedron where
  edges : Fin 6 → Rod

/-- A function to count distinct tetrahedral edge frameworks -/
noncomputable def count_distinct_tetrahedra (sr : SixRods) : ℕ := sorry

/-- The main theorem -/
theorem distinct_tetrahedra_count (sr : SixRods) 
  (h : can_form_triangle sr) : 
  count_distinct_tetrahedra sr = 30 := by sorry

end NUMINAMATH_CALUDE_distinct_tetrahedra_count_l3426_342697


namespace NUMINAMATH_CALUDE_sandcastle_problem_l3426_342622

theorem sandcastle_problem (mark_castles : ℕ) : 
  (mark_castles * 10 + mark_castles) +  -- Mark's castles and towers
  ((3 * mark_castles) * 5 + (3 * mark_castles)) = 580 -- Jeff's castles and towers
  → mark_castles = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_problem_l3426_342622


namespace NUMINAMATH_CALUDE_reading_speed_first_half_l3426_342662

/-- Given a book with specific reading conditions, calculate the reading speed for the first half. -/
theorem reading_speed_first_half (total_pages : ℕ) (second_half_speed : ℕ) (total_days : ℕ) : 
  total_pages = 500 → 
  second_half_speed = 5 → 
  total_days = 75 → 
  (total_pages / 2) / (total_days - (total_pages / 2) / second_half_speed) = 10 := by
  sorry

#check reading_speed_first_half

end NUMINAMATH_CALUDE_reading_speed_first_half_l3426_342662


namespace NUMINAMATH_CALUDE_yellow_hats_count_l3426_342647

/-- The number of yellow hard hats initially in the truck -/
def yellow_hats : ℕ := 24

/-- The initial number of pink hard hats -/
def initial_pink : ℕ := 26

/-- The initial number of green hard hats -/
def initial_green : ℕ := 15

/-- The number of pink hard hats Carl takes -/
def carl_pink : ℕ := 4

/-- The number of pink hard hats John takes -/
def john_pink : ℕ := 6

/-- The number of green hard hats John takes -/
def john_green : ℕ := 2 * john_pink

/-- The total number of hard hats remaining after Carl and John take theirs -/
def total_remaining : ℕ := 43

theorem yellow_hats_count :
  yellow_hats = total_remaining - (initial_pink - carl_pink - john_pink) - (initial_green - john_green) :=
by sorry

end NUMINAMATH_CALUDE_yellow_hats_count_l3426_342647


namespace NUMINAMATH_CALUDE_container_capacity_l3426_342607

theorem container_capacity (x : ℝ) 
  (h1 : (1/4) * x + 300 = (3/4) * x) : x = 600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3426_342607


namespace NUMINAMATH_CALUDE_find_b_l3426_342636

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l3426_342636


namespace NUMINAMATH_CALUDE_lucy_grocery_cost_l3426_342608

/-- Represents the total cost of Lucy's grocery purchases in USD -/
def total_cost_usd (cookies_packs : ℕ) (cookies_price : ℚ)
                   (noodles_packs : ℕ) (noodles_price : ℚ)
                   (soup_cans : ℕ) (soup_price : ℚ)
                   (cereals_boxes : ℕ) (cereals_price : ℚ)
                   (crackers_packs : ℕ) (crackers_price : ℚ)
                   (usd_to_eur : ℚ) (usd_to_gbp : ℚ) : ℚ :=
  cookies_packs * cookies_price +
  (noodles_packs * noodles_price) / usd_to_eur +
  (soup_cans * soup_price) / usd_to_gbp +
  cereals_boxes * cereals_price +
  (crackers_packs * crackers_price) / usd_to_eur

/-- The theorem stating that Lucy's total grocery cost is $183.92 -/
theorem lucy_grocery_cost :
  total_cost_usd 12 (5/2) 16 (9/5) 28 (6/5) 5 (17/5) 45 (11/10) (17/20) (3/4) = 18392/100 := by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_cost_l3426_342608


namespace NUMINAMATH_CALUDE_area_of_region_l3426_342679

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 - 4 * x - x * y + 2 * y = 0

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2 ∧ 0 ≤ p.2}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l3426_342679


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3426_342672

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/3))^(1/4))^3 * (((b^16)^(1/4))^(1/3))^3 = b^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3426_342672


namespace NUMINAMATH_CALUDE_sixth_year_fee_l3426_342624

def membership_fee (initial_fee : ℕ) (yearly_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * yearly_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end NUMINAMATH_CALUDE_sixth_year_fee_l3426_342624


namespace NUMINAMATH_CALUDE_triangle_proof_l3426_342676

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0 →
  (A = π / 3 ∧ 
   (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l3426_342676


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l3426_342695

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ :=
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice * prob_one * prob_not_one ^ (num_dice - 1)) +
  2 * (num_dice * (num_dice - 1) / 2 * prob_one ^ 2 * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l3426_342695


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3426_342680

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The condition "x = 2" for the given vectors -/
def condition (x : ℝ) : Prop := x = 2

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, condition x → are_parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, are_parallel (a x) (b x) → condition x) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3426_342680


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l3426_342637

def probability_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem mayor_approval_probability : 
  probability_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l3426_342637


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3426_342623

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x + 11) - (x^6 + 2 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 5 * x^4 - x^3 + x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3426_342623


namespace NUMINAMATH_CALUDE_triangle_special_angles_l3426_342655

theorem triangle_special_angles (A B C : ℝ) (a b c : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Angles are positive
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  C = 2 * A ∧              -- Angle C is twice angle A
  b = 2 * a ∧              -- Side b is twice side a
  a * Real.sin B = b * Real.sin A ∧  -- Law of sines
  a * Real.sin C = c * Real.sin A ∧  -- Law of sines
  a^2 + b^2 = c^2          -- Pythagorean theorem
  →
  A = Real.pi / 6 ∧ B = Real.pi / 2 ∧ C = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_angles_l3426_342655


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_l3426_342660

/-- The number of cartons of blueberries Mason has -/
def num_cartons : ℕ := 8

/-- The number of blueberries in each carton -/
def blueberries_per_carton : ℕ := 300

/-- The number of blueberries used per muffin -/
def blueberries_per_muffin : ℕ := 18

/-- The number of blueberries left after making blueberry muffins -/
def blueberries_left : ℕ := 54

/-- The number of cinnamon muffins made -/
def cinnamon_muffins : ℕ := 80

/-- The number of chocolate muffins made -/
def chocolate_muffins : ℕ := 40

/-- The number of cranberry muffins made -/
def cranberry_muffins : ℕ := 50

/-- The number of lemon muffins made -/
def lemon_muffins : ℕ := 30

/-- Theorem stating that the percentage of blueberry muffins is approximately 39.39% -/
theorem blueberry_muffin_percentage :
  let total_blueberries := num_cartons * blueberries_per_carton
  let used_blueberries := total_blueberries - blueberries_left
  let blueberry_muffins := used_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins + chocolate_muffins + cranberry_muffins + lemon_muffins
  let percentage := (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100
  abs (percentage - 39.39) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_l3426_342660


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3426_342644

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3426_342644


namespace NUMINAMATH_CALUDE_repeating_decimal_incorrect_expression_l3426_342694

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The theorem stating that the given expression is not always true for repeating decimals -/
theorem repeating_decimal_incorrect_expression (D : RepeatingDecimal) :
  ¬ (∀ (D : RepeatingDecimal), 10^D.r * (10^D.s - 1) * (D.P / 10^D.r + D.Q / (10^D.r * (10^D.s - 1))) = D.Q * (D.P - 1)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_incorrect_expression_l3426_342694


namespace NUMINAMATH_CALUDE_product_of_fractions_l3426_342621

theorem product_of_fractions : 
  (7 : ℚ) / 4 * 14 / 35 * 21 / 12 * 28 / 56 * 49 / 28 * 42 / 84 * 63 / 36 * 56 / 112 = 1201 / 12800 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3426_342621


namespace NUMINAMATH_CALUDE_tims_balloons_l3426_342656

def dans_balloons : ℝ := 29.0
def dans_multiple : ℝ := 7.0

theorem tims_balloons : ⌊dans_balloons / dans_multiple⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_tims_balloons_l3426_342656


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l3426_342626

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 18/5

/-- The amount John had left after spending at the arcade and toy store, in dollars -/
def remaining_amount : ℚ := 24/25

theorem johns_arcade_spending :
  let remaining_after_arcade : ℚ := weekly_allowance * (1 - arcade_fraction)
  let spent_at_toy_store : ℚ := remaining_after_arcade * (1/3)
  remaining_after_arcade - spent_at_toy_store = remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l3426_342626


namespace NUMINAMATH_CALUDE_principal_calculation_l3426_342614

/-- Proves that given specific conditions, the principal amount is 900 --/
theorem principal_calculation (interest_rate : ℚ) (time : ℚ) (final_amount : ℚ) :
  interest_rate = 5 / 100 →
  time = 12 / 5 →
  final_amount = 1008 →
  final_amount = (1 + interest_rate * time) * 900 :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l3426_342614


namespace NUMINAMATH_CALUDE_inverse_matrix_part1_curve_transformation_part2_l3426_342651

def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; b, 1]

theorem inverse_matrix_part1 :
  let M := M 2 3
  M⁻¹ = !![(-1/5 : ℝ), 2/5; 3/5, -1/5] := by sorry

theorem curve_transformation_part2 (a b : ℝ) :
  (∀ x y : ℝ, x^2 + 4*x*y + 2*y^2 = 1 →
    let x' := x + a*y
    let y' := b*x + y
    x'^2 - 2*y'^2 = 1) →
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_matrix_part1_curve_transformation_part2_l3426_342651


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3426_342604

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |(-a)| = |(-1)|) 
  (h2 : b^2 = 9)
  (h3 : |a - b| = b - a) : 
  a + b = 2 ∨ a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3426_342604


namespace NUMINAMATH_CALUDE_no_real_roots_equation_3_l3426_342663

theorem no_real_roots_equation_3 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (geom_seq : b^2 = a*c)
  (real_roots_1 : a^2 ≥ 4)
  (no_real_roots_2 : b^2 < 8) :
  c^2 < 16 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_equation_3_l3426_342663


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l3426_342635

theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ) :
  total_balls = 11 →
  white_balls = 5 →
  drawn_balls = 5 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 462 := by
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l3426_342635


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3426_342698

/-- Predicate to check if the equation ax^2 + by^2 = 1 represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that ab > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) := by sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3426_342698


namespace NUMINAMATH_CALUDE_rope_ratio_l3426_342699

/-- Given a rope of 35 inches cut into two pieces, with the longer piece being 20 inches,
    prove that the ratio of the longer piece to the shorter piece is 4:3. -/
theorem rope_ratio : ∃ (a b : ℕ), a = 4 ∧ b = 3 ∧ 20 * b = 15 * a := by
  sorry

end NUMINAMATH_CALUDE_rope_ratio_l3426_342699


namespace NUMINAMATH_CALUDE_another_hamiltonian_cycle_l3426_342629

/-- A graph with n vertices where each vertex has exactly 3 neighbors -/
structure ThreeRegularGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  degree_three : ∀ v : Fin n, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- A Hamiltonian cycle in a graph -/
def HamiltonianCycle {n : ℕ} (G : ThreeRegularGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.toFinset = G.vertices }

/-- Two Hamiltonian cycles are equivalent if one can be obtained from the other by rotation or reflection -/
def EquivalentCycles {n : ℕ} (G : ThreeRegularGraph n) (c1 c2 : HamiltonianCycle G) : Prop :=
  ∃ (k : ℕ) (reflect : Bool),
    c2.val = if reflect then c1.val.reverse.rotateRight k else c1.val.rotateRight k

theorem another_hamiltonian_cycle {n : ℕ} (G : ThreeRegularGraph n) (c : HamiltonianCycle G) :
  ∃ (c' : HamiltonianCycle G), ¬EquivalentCycles G c c' :=
sorry

end NUMINAMATH_CALUDE_another_hamiltonian_cycle_l3426_342629


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3426_342634

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3426_342634


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l3426_342674

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLength : ℕ := 4

theorem last_four_digits_of_5_to_2011 :
  lastFourDigits 2011 = lastFourDigits 7 :=
by sorry

theorem last_four_digits_of_5_to_7 :
  lastFourDigits 7 = 8125 :=
by sorry

theorem last_four_digits_of_5_to_2011_is_8125 :
  lastFourDigits 2011 = 8125 :=
by sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l3426_342674


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3426_342601

theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -1]
  5 • v1 - 3 • v2 + v3 = ![20, -44] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3426_342601


namespace NUMINAMATH_CALUDE_sandwiches_per_student_l3426_342640

theorem sandwiches_per_student
  (students_per_group : ℕ)
  (total_groups : ℕ)
  (total_bread_pieces : ℕ)
  (bread_per_sandwich : ℕ)
  (h1 : students_per_group = 6)
  (h2 : total_groups = 5)
  (h3 : total_bread_pieces = 120)
  (h4 : bread_per_sandwich = 2) :
  total_bread_pieces / (bread_per_sandwich * (students_per_group * total_groups)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sandwiches_per_student_l3426_342640


namespace NUMINAMATH_CALUDE_school_store_pricing_l3426_342657

/-- Given the cost of pencils and notebooks in a school store, 
    calculate the cost of a specific combination. -/
theorem school_store_pricing 
  (pencil_cost notebook_cost : ℚ) 
  (h1 : 6 * pencil_cost + 6 * notebook_cost = 390/100)
  (h2 : 8 * pencil_cost + 4 * notebook_cost = 328/100) : 
  20 * pencil_cost + 14 * notebook_cost = 1012/100 := by
  sorry

end NUMINAMATH_CALUDE_school_store_pricing_l3426_342657


namespace NUMINAMATH_CALUDE_selection_problem_l3426_342613

theorem selection_problem (n_teachers : ℕ) (n_students : ℕ) : n_teachers = 4 → n_students = 5 →
  (Nat.choose n_teachers 1 * Nat.choose n_students 2 + 
   Nat.choose n_teachers 2 * Nat.choose n_students 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l3426_342613


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l3426_342671

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

theorem imaginary_part_of_product :
  (complex_mul 2 1 1 (-3)).im = -5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l3426_342671


namespace NUMINAMATH_CALUDE_suraj_average_after_ninth_innings_l3426_342670

/-- Represents a cricket player's performance -/
structure CricketPerformance where
  innings : ℕ
  lowestScore : ℕ
  highestScore : ℕ
  fiftyPlusInnings : ℕ
  totalRuns : ℕ

/-- Calculates the average runs per innings -/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

theorem suraj_average_after_ninth_innings 
  (suraj : CricketPerformance)
  (h1 : suraj.innings = 8)
  (h2 : suraj.lowestScore = 25)
  (h3 : suraj.highestScore = 80)
  (h4 : suraj.fiftyPlusInnings = 3)
  (h5 : average suraj + 6 = average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 }) :
  average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 } = 42 := by
  sorry


end NUMINAMATH_CALUDE_suraj_average_after_ninth_innings_l3426_342670


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3426_342646

/-- Given a point P in the second quadrant, prove that P₁ is in the third quadrant -/
theorem point_in_third_quadrant (a b : ℝ) (h : a < 0 ∧ b > 0) :
  -b < 0 ∧ a - 1 < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3426_342646


namespace NUMINAMATH_CALUDE_residue_mod_32_l3426_342643

theorem residue_mod_32 : Int.mod (-1277) 32 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_32_l3426_342643


namespace NUMINAMATH_CALUDE_smallest_number_l3426_342609

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 2) (hc : c = -4) (hd : d = -1) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3426_342609


namespace NUMINAMATH_CALUDE_stone_162_is_12_l3426_342686

/-- The number of stones in the circular arrangement -/
def n : ℕ := 15

/-- The count we're interested in -/
def target_count : ℕ := 162

/-- The function that maps a count to its corresponding stone number -/
def stone_number (count : ℕ) : ℕ := 
  if count % n = 0 then n else count % n

theorem stone_162_is_12 : stone_number target_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_stone_162_is_12_l3426_342686


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l3426_342616

theorem shoe_price_calculation (thursday_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  thursday_price = 50 →
  friday_increase = 0.2 →
  monday_decrease = 0.15 →
  thursday_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry


end NUMINAMATH_CALUDE_shoe_price_calculation_l3426_342616


namespace NUMINAMATH_CALUDE_unique_solution_l3426_342628

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem unique_solution : 
  ∃! x : ℕ, 
    digit_product x = 44 * x - 86868 ∧ 
    is_perfect_cube (digit_sum x) ∧
    x = 1989 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3426_342628


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l3426_342625

theorem sum_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^8 = a*x^12 + a₁*x^11 + a₂*x^10 + a₃*x^9 + a₄*x^8 + 
    a₅*x^7 + a₆*x^6 + a₇*x^5 + a₈*x^4 + a₉*x^3 + a₁₀*x^2 + a₁₁*x + a₁₂) →
  a = 1 →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l3426_342625


namespace NUMINAMATH_CALUDE_equation_solutions_l3426_342631

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x * y + y * z + z * x = 2 * (x + y + z)} =
  {(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (2, 2, 2), (4, 1, 2), (4, 2, 1)} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3426_342631


namespace NUMINAMATH_CALUDE_f_zero_equals_two_l3426_342688

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = f x₁ + f x₂ + f x₃ + f x₄ + f x₅ - 8

theorem f_zero_equals_two (f : ℝ → ℝ) (h : f_property f) : f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_equals_two_l3426_342688


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l3426_342683

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ 3} := by sorry

-- Theorem for (A ∩ B)ᶜ
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 4 ∨ x ≥ 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l3426_342683


namespace NUMINAMATH_CALUDE_coin_collection_problem_l3426_342682

theorem coin_collection_problem (n d q : ℕ) : 
  n + d + q = 23 →
  5 * n + 10 * d + 25 * q = 320 →
  d = n + 3 →
  q - n = 2 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l3426_342682


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l3426_342689

def number_of_dogs : ℕ := 10
def group_sizes : List ℕ := [3, 5, 2]

theorem dog_grouping_theorem :
  let remaining_dogs := number_of_dogs - 2  -- Fluffy and Nipper are pre-placed
  let ways_to_fill_fluffy_group := Nat.choose remaining_dogs (group_sizes[0] - 1)
  let remaining_after_fluffy := remaining_dogs - (group_sizes[0] - 1)
  let ways_to_fill_nipper_group := Nat.choose remaining_after_fluffy (group_sizes[1] - 1)
  ways_to_fill_fluffy_group * ways_to_fill_nipper_group = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l3426_342689


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l3426_342603

/-- Given a cylinder with height 5 inches and radius r, 
    if increasing the radius by 4 inches or increasing the height by 4 inches 
    results in the same volume, then r = 5 + 3√5 -/
theorem cylinder_radius_problem (r : ℝ) : 
  (π * (r + 4)^2 * 5 = π * r^2 * 9) → r = 5 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l3426_342603


namespace NUMINAMATH_CALUDE_calorie_difference_per_dollar_l3426_342668

/-- Calculates the difference in calories per dollar between burgers and burritos -/
theorem calorie_difference_per_dollar : 
  let burrito_count : ℕ := 10
  let burrito_price : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_price : ℚ := 8
  let burger_calories : ℕ := 400
  let burrito_calories_per_dollar := (burrito_count * burrito_calories : ℚ) / burrito_price
  let burger_calories_per_dollar := (burger_count * burger_calories : ℚ) / burger_price
  burger_calories_per_dollar - burrito_calories_per_dollar = 50
:= by sorry


end NUMINAMATH_CALUDE_calorie_difference_per_dollar_l3426_342668


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3426_342684

-- Define the sets P and M
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- Define what it means for a condition to be sufficient
def is_sufficient (a : ℝ) : Prop := P ⊆ M a

-- Define what it means for a condition to be necessary
def is_necessary (a : ℝ) : Prop := ∀ b : ℝ, P ⊆ M b → a ≤ b

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (is_sufficient 1) ∧ ¬(is_necessary 1) := by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3426_342684


namespace NUMINAMATH_CALUDE_side_margin_width_l3426_342649

/-- Given a sheet of paper with dimensions and margin constraints, prove the side margin width. -/
theorem side_margin_width (sheet_width sheet_length top_bottom_margin : ℝ)
  (typing_area_percentage : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : top_bottom_margin = 3)
  (h4 : typing_area_percentage = 0.64) :
  ∃ (side_margin : ℝ),
    side_margin = 2 ∧
    (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) =
      typing_area_percentage * sheet_width * sheet_length :=
by sorry

end NUMINAMATH_CALUDE_side_margin_width_l3426_342649


namespace NUMINAMATH_CALUDE_ball_problem_l3426_342691

/-- The number of red balls in the bag -/
def red_balls (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls in the bag -/
def yellow_balls (a : ℕ) : ℕ := a

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := red_balls a + yellow_balls a + blue_balls

/-- The score for drawing a red ball -/
def red_score : ℕ := 1

/-- The score for drawing a yellow ball -/
def yellow_score : ℕ := 2

/-- The score for drawing a blue ball -/
def blue_score : ℕ := 3

/-- The expected value of the score when drawing a ball -/
def expected_value (a : ℕ) : ℚ :=
  (red_score * red_balls a + yellow_score * yellow_balls a + blue_score * blue_balls) / total_balls a

theorem ball_problem (a : ℕ) (h1 : a > 0) (h2 : expected_value a = 5/3) :
  a = 2 ∧ (Nat.choose (red_balls a) 1 * Nat.choose (yellow_balls a) 2 +
           Nat.choose (red_balls a) 2 * Nat.choose blue_balls 1) /
          Nat.choose (total_balls a) 3 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ball_problem_l3426_342691


namespace NUMINAMATH_CALUDE_function_inequality_l3426_342633

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_ab : a > b ∧ b > 1) 
  (h_deriv : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f a + f b ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3426_342633


namespace NUMINAMATH_CALUDE_sister_ages_l3426_342675

theorem sister_ages (x y : ℕ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 988) : x = 11 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_ages_l3426_342675


namespace NUMINAMATH_CALUDE_roger_cookie_price_l3426_342669

/-- Represents a cookie batch -/
structure CookieBatch where
  shape : String
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem roger_cookie_price (art_batch roger_batch : CookieBatch) 
  (h1 : art_batch.shape = "rectangle")
  (h2 : roger_batch.shape = "square")
  (h3 : art_batch.count = 15)
  (h4 : roger_batch.count = 20)
  (h5 : art_batch.price = 75/100)
  (h6 : totalEarnings art_batch = totalEarnings roger_batch) :
  roger_batch.price = 5625/10000 := by
  sorry

#eval (5625 : ℚ) / 10000  -- Expected output: 0.5625

end NUMINAMATH_CALUDE_roger_cookie_price_l3426_342669


namespace NUMINAMATH_CALUDE_prob_one_letter_each_name_l3426_342638

/-- Probability of selecting one letter from each person's name -/
theorem prob_one_letter_each_name :
  let total_cards : ℕ := 14
  let elena_cards : ℕ := 5
  let mark_cards : ℕ := 4
  let julia_cards : ℕ := 5
  let num_permutations : ℕ := 6  -- 3! permutations of 3 items
  
  elena_cards + mark_cards + julia_cards = total_cards →
  
  (elena_cards : ℚ) / total_cards *
  (mark_cards : ℚ) / (total_cards - 1) *
  (julia_cards : ℚ) / (total_cards - 2) *
  num_permutations = 25 / 91 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_letter_each_name_l3426_342638


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3426_342685

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x + 1) - 2 / (x - 1) = 0) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3426_342685
