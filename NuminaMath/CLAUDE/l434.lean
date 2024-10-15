import Mathlib

namespace NUMINAMATH_CALUDE_total_days_1999_to_2005_l434_43497

def is_leap_year (year : ℕ) : Bool :=
  year = 2000 || year = 2004

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).map (fun i => days_in_year (start_year + i))
    |>.sum

theorem total_days_1999_to_2005 :
  total_days 1999 2005 = 2557 := by
  sorry

end NUMINAMATH_CALUDE_total_days_1999_to_2005_l434_43497


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l434_43484

/-- The profit percentage for a merchant who marks up goods by 75% and then offers a 10% discount -/
theorem merchant_profit_percentage : 
  let markup_percentage : ℝ := 75
  let discount_percentage : ℝ := 10
  let cost_price : ℝ := 100
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 57.5 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l434_43484


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l434_43452

/-- Given an inequality a ≤ 3x + 5 ≤ b, where the length of the interval of solutions is 15, prove that b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 3*x + 5 ∧ 3*x + 5 ≤ b) → 
  ((b - 5) / 3 - (a - 5) / 3 = 15) → 
  b - a = 45 := by sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l434_43452


namespace NUMINAMATH_CALUDE_book_code_is_mirror_l434_43420

/-- Represents the coding system --/
structure CodeSystem where
  book : String
  mirror : String
  board : String
  writing_item : String

/-- The given coding rules --/
def given_code : CodeSystem :=
  { book := "certain_item",
    mirror := "board",
    board := "board",
    writing_item := "2" }

/-- Theorem: The code for 'book' is 'mirror' --/
theorem book_code_is_mirror (code : CodeSystem) (h1 : code.book = "certain_item") 
  (h2 : code.mirror = "board") : code.book = "mirror" :=
by sorry

end NUMINAMATH_CALUDE_book_code_is_mirror_l434_43420


namespace NUMINAMATH_CALUDE_B_power_48_l434_43454

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_48 : 
  B ^ 48 = !![0, 0, 0; 0, 1, 0; 0, 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_48_l434_43454


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l434_43479

theorem neither_necessary_nor_sufficient : 
  ¬(∀ x : ℝ, -1/2 < x ∧ x < 1 → 0 < x ∧ x < 2) ∧ 
  ¬(∀ x : ℝ, 0 < x ∧ x < 2 → -1/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l434_43479


namespace NUMINAMATH_CALUDE_sammy_gift_wrapping_l434_43450

/-- The number of gifts Sammy can wrap -/
def num_gifts : ℕ := 8

/-- The length of ribbon required for each gift in meters -/
def ribbon_per_gift : ℝ := 1.5

/-- The total length of Tom's ribbon in meters -/
def total_ribbon : ℝ := 15

/-- The length of ribbon left after wrapping all gifts in meters -/
def ribbon_left : ℝ := 3

/-- Theorem stating that the number of gifts Sammy can wrap is correct -/
theorem sammy_gift_wrapping :
  (↑num_gifts : ℝ) * ribbon_per_gift = total_ribbon - ribbon_left :=
by sorry

end NUMINAMATH_CALUDE_sammy_gift_wrapping_l434_43450


namespace NUMINAMATH_CALUDE_gcd_5040_13860_l434_43401

theorem gcd_5040_13860 : Nat.gcd 5040 13860 = 420 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5040_13860_l434_43401


namespace NUMINAMATH_CALUDE_distance_between_ports_l434_43424

/-- The distance between two ports given ship and current speeds and time difference -/
theorem distance_between_ports (ship_speed : ℝ) (current_speed : ℝ) (time_diff : ℝ) :
  ship_speed > current_speed →
  ship_speed = 24 →
  current_speed = 3 →
  time_diff = 5 →
  ∃ (distance : ℝ),
    distance / (ship_speed - current_speed) - distance / (ship_speed + current_speed) = time_diff ∧
    distance = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_ports_l434_43424


namespace NUMINAMATH_CALUDE_outfit_count_l434_43466

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 8

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- A function that calculates the number of valid outfits -/
def valid_outfits : ℕ := 
  num_colors * num_colors * num_colors - 
  (num_colors * (num_colors - 1) * 3)

/-- Theorem stating that the number of valid outfits is 344 -/
theorem outfit_count : valid_outfits = 344 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l434_43466


namespace NUMINAMATH_CALUDE_smallest_divisible_by_nine_l434_43482

/-- The smallest digit d such that 528,d46 is divisible by 9 -/
def smallest_digit : ℕ := 2

/-- A function that constructs the number 528,d46 given a digit d -/
def construct_number (d : ℕ) : ℕ := 528000 + d * 100 + 46

theorem smallest_divisible_by_nine :
  (∀ d : ℕ, d < smallest_digit → ¬(9 ∣ construct_number d)) ∧
  (9 ∣ construct_number smallest_digit) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_nine_l434_43482


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l434_43488

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The set of three-digit numbers -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_7_heavy : 
  ∃ (n : ℕ), n ∈ three_digit_numbers ∧ is_7_heavy n ∧ 
  ∀ (m : ℕ), m ∈ three_digit_numbers → is_7_heavy m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l434_43488


namespace NUMINAMATH_CALUDE_square_triangle_count_l434_43451

theorem square_triangle_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h_total_shapes : total_shapes = 35)
  (h_total_edges : total_edges = 120) :
  ∃ (squares triangles : ℕ),
    squares + triangles = total_shapes ∧
    4 * squares + 3 * triangles = total_edges ∧
    squares = 20 ∧
    triangles = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_count_l434_43451


namespace NUMINAMATH_CALUDE_zongzi_price_proof_l434_43405

-- Define the unit price of type B zongzi
def unit_price_B : ℝ := 4

-- Define the conditions
def amount_A : ℝ := 1200
def amount_B : ℝ := 800
def quantity_difference : ℕ := 50

-- Theorem statement
theorem zongzi_price_proof :
  -- Conditions
  (amount_A = (2 * unit_price_B) * ((amount_B / unit_price_B) - quantity_difference)) ∧
  (amount_B = unit_price_B * (amount_B / unit_price_B)) →
  -- Conclusion
  unit_price_B = 4 := by
  sorry


end NUMINAMATH_CALUDE_zongzi_price_proof_l434_43405


namespace NUMINAMATH_CALUDE_mopping_time_is_30_l434_43444

def vacuum_time : ℕ := 45
def dusting_time : ℕ := 60
def cat_brushing_time : ℕ := 5
def num_cats : ℕ := 3
def total_free_time : ℕ := 3 * 60
def remaining_free_time : ℕ := 30

def total_cleaning_time : ℕ := total_free_time - remaining_free_time

def other_tasks_time : ℕ := vacuum_time + dusting_time + (cat_brushing_time * num_cats)

theorem mopping_time_is_30 : 
  total_cleaning_time - other_tasks_time = 30 := by sorry

end NUMINAMATH_CALUDE_mopping_time_is_30_l434_43444


namespace NUMINAMATH_CALUDE_m_values_l434_43457

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem m_values : ∀ m : ℝ, (A ∪ B m = A) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/5) := by sorry

end NUMINAMATH_CALUDE_m_values_l434_43457


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l434_43469

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (d : ℚ) (h1 : a = 1/2) (h2 : d = 2/3) :
  arithmetic_sequence a d 10 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l434_43469


namespace NUMINAMATH_CALUDE_vector_lines_correct_l434_43402

/-- Vector field in R³ -/
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (0, 9*z, -4*y)

/-- Vector lines of the given vector field -/
def vector_lines (x y z C₁ C₂ : ℝ) : Prop :=
  9 * z^2 + 4 * y^2 = C₁ ∧ x = C₂

/-- Theorem stating that the vector lines are correct for the given vector field -/
theorem vector_lines_correct :
  ∀ (x y z C₁ C₂ : ℝ),
    vector_lines x y z C₁ C₂ ↔
    ∃ (t : ℝ), (x, y, z) = (C₂, 
                            9 * t * (vector_field x y z).2.1, 
                            -4 * t * (vector_field x y z).2.2) :=
sorry

end NUMINAMATH_CALUDE_vector_lines_correct_l434_43402


namespace NUMINAMATH_CALUDE_probability_circle_or_square_l434_43441

def total_figures : ℕ := 10
def num_circles : ℕ := 3
def num_squares : ℕ := 4
def num_triangles : ℕ := 3

theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_circle_or_square_l434_43441


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l434_43491

/-- An isosceles triangle with two sides of length 7 and one side of length 3 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 3 → -- Two sides are 7cm and one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) → -- Isosceles condition
  a + b + c = 17 := by -- Perimeter is 17cm
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l434_43491


namespace NUMINAMATH_CALUDE_inequality_of_powers_l434_43406

theorem inequality_of_powers (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l434_43406


namespace NUMINAMATH_CALUDE_digit_arrangement_count_l434_43473

theorem digit_arrangement_count : 
  let digits : List ℕ := [4, 7, 5, 2, 0]
  let n : ℕ := digits.length
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  96 = (n - 1) * Nat.factorial (non_zero_digits.length) := by
  sorry

end NUMINAMATH_CALUDE_digit_arrangement_count_l434_43473


namespace NUMINAMATH_CALUDE_factory_employees_count_l434_43411

/-- Represents the profit calculation for a t-shirt factory --/
def factory_profit (num_employees : ℕ) : ℚ :=
  let shirts_per_employee := 20
  let shirt_price := 35
  let hourly_wage := 12
  let per_shirt_bonus := 5
  let hours_per_shift := 8
  let nonemployee_expenses := 1000
  let total_shirts := num_employees * shirts_per_employee
  let revenue := total_shirts * shirt_price
  let employee_pay := num_employees * (hourly_wage * hours_per_shift + per_shirt_bonus * shirts_per_employee)
  revenue - employee_pay - nonemployee_expenses

/-- The number of employees that results in the given profit --/
theorem factory_employees_count : 
  ∃ (n : ℕ), factory_profit n = 9080 ∧ n = 20 := by
  sorry


end NUMINAMATH_CALUDE_factory_employees_count_l434_43411


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l434_43437

/-- Proves that the initial volume of a mixture is 425 litres given the conditions -/
theorem initial_mixture_volume :
  ∀ (V : ℝ),
  (V > 0) →
  (0.10 * V = V * 0.10) →
  (0.10 * V + 25 = 0.15 * (V + 25)) →
  V = 425 :=
λ V hV_pos hWater_ratio hNew_ratio =>
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l434_43437


namespace NUMINAMATH_CALUDE_solution_set_equality_l434_43407

theorem solution_set_equality : 
  {x : ℝ | 1 < |x + 2| ∧ |x + 2| < 5} = 
  {x : ℝ | -7 < x ∧ x < -3} ∪ {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l434_43407


namespace NUMINAMATH_CALUDE_boys_camp_total_l434_43459

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 21 →
  total = 150 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l434_43459


namespace NUMINAMATH_CALUDE_sine_unit_implies_on_y_axis_l434_43476

-- Define the type for angles
def Angle : Type := ℝ

-- Define the sine function
noncomputable def sine (α : Angle) : ℝ := Real.sin α

-- Define a predicate for a directed line segment of unit length
def is_unit_directed_segment (x : ℝ) : Prop := x = 1 ∨ x = -1

-- Define a predicate for a point being on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem sine_unit_implies_on_y_axis (α : Angle) :
  is_unit_directed_segment (sine α) →
  ∃ (y : ℝ), on_y_axis 0 y ∧ (0, y) = (Real.cos α, Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_sine_unit_implies_on_y_axis_l434_43476


namespace NUMINAMATH_CALUDE_female_employees_count_l434_43470

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  female : ℕ
  male : ℕ
  female_managers : ℕ
  male_managers : ℕ

/-- The conditions of the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 300 ∧
  c.female_managers + c.male_managers = (2 : ℚ) / 5 * c.total ∧
  c.male_managers = (2 : ℚ) / 5 * c.male ∧
  c.total = c.female + c.male

/-- The theorem stating that the number of female employees is 750 -/
theorem female_employees_count (c : Company) : 
  company_conditions c → c.female = 750 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l434_43470


namespace NUMINAMATH_CALUDE_equation_solution_l434_43461

theorem equation_solution : ∃! x : ℚ, (3 / 4 : ℚ) + 1 / x = 7 / 8 :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l434_43461


namespace NUMINAMATH_CALUDE_square_side_length_l434_43422

/-- Given a square with diagonal length 2√2, prove that its side length is 2. -/
theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = (d * d) / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l434_43422


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l434_43416

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x = -y → x^2 - y^2 - x - y = 0) ∧ 
  ¬(x^2 - y^2 - x - y = 0 → x = -y) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l434_43416


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l434_43487

/-- Calculates the number of light bulb packs needed given the number of bulbs required for each room --/
def calculate_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let other_rooms_total := bedroom + bathroom + kitchen + basement
  let garage := other_rooms_total / 2
  let total_bulbs := other_rooms_total + garage
  (total_bulbs + 1) / 2

/-- Proves that Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  calculate_packs_needed 2 1 1 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l434_43487


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l434_43436

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio := 
  let f := standard_ratio.flavoring
  let c := standard_ratio.corn_syrup
  let w := standard_ratio.water
  ⟨f, f * 4, f * 15⟩

/-- Calculates the amount of water given the amount of corn syrup -/
def water_amount (corn_syrup_amount : ℚ) : ℚ :=
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup

/-- Theorem: The amount of water in the sport formulation is 7.5 ounces when there are 2 ounces of corn syrup -/
theorem water_amount_in_sport_formulation :
  water_amount 2 = 7.5 := by sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l434_43436


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l434_43492

theorem cement_mixture_weight : 
  ∀ W : ℝ, 
    (5/14 + 3/10 + 2/9 + 1/7) * W + 2.5 = W → 
    W = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l434_43492


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l434_43496

theorem quadratic_equation_m_value : ∃! m : ℤ, |m| = 2 ∧ m + 2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l434_43496


namespace NUMINAMATH_CALUDE_family_reunion_children_l434_43489

theorem family_reunion_children (adults children : ℕ) : 
  adults = children / 3 →
  adults / 3 + 10 = adults →
  children = 45 := by
sorry

end NUMINAMATH_CALUDE_family_reunion_children_l434_43489


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_perimeter_twice_area_l434_43460

theorem isosceles_right_triangle_perimeter_twice_area :
  ∃! a : ℝ, a > 0 ∧ (2 * a + a * Real.sqrt 2 = 2 * (1 / 2 * a^2)) := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_perimeter_twice_area_l434_43460


namespace NUMINAMATH_CALUDE_student_tickets_sold_l434_43417

theorem student_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    student_tickets = 410 := by
  sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l434_43417


namespace NUMINAMATH_CALUDE_fish_count_after_transfer_l434_43456

/-- The total number of fish after Lilly gives some to Jack -/
def total_fish (lilly_initial : ℕ) (rosy : ℕ) (jack_initial : ℕ) (transfer : ℕ) : ℕ :=
  (lilly_initial - transfer) + rosy + (jack_initial + transfer)

/-- Theorem stating the total number of fish after the transfer -/
theorem fish_count_after_transfer :
  total_fish 10 9 15 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_after_transfer_l434_43456


namespace NUMINAMATH_CALUDE_max_value_polynomial_l434_43472

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ M : ℝ, M = (6084 : ℝ) / 17 ∧
  ∀ z w : ℝ, z + w = 5 →
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ M ∧
    ∃ a b : ℝ, a + b = 5 ∧
      a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l434_43472


namespace NUMINAMATH_CALUDE_oranges_per_box_l434_43468

/-- Given 45 oranges and 9 boxes, prove that the number of oranges per box is 5 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 45) (h2 : num_boxes = 9) : 
  total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l434_43468


namespace NUMINAMATH_CALUDE_area_above_line_ratio_l434_43439

/-- Given two positive real numbers a and b, where a > b > 1/2 * a,
    and two squares with side lengths a and b placed next to each other,
    with the larger square having its lower left corner at (0,0) and
    the smaller square having its lower left corner at (a,0),
    if the area above the line passing through (0,a) and (a+b,0) in both squares is 2013,
    and (a,b) is the unique pair maximizing a+b,
    then a/b = ∛5√3. -/
theorem area_above_line_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hba : b > (1/2) * a) (harea : (a^3 / (2*(a+b))) + (a*b/2) = 2013)
  (hmax : ∀ (x y : ℝ), x > 0 → y > 0 → x > y → y > (1/2) * x →
    (x^3 / (2*(x+y))) + (x*y/2) = 2013 → x + y ≤ a + b) :
  a / b = (3 : ℝ)^(1/5) :=
sorry

end NUMINAMATH_CALUDE_area_above_line_ratio_l434_43439


namespace NUMINAMATH_CALUDE_tan_3285_degrees_l434_43425

theorem tan_3285_degrees : Real.tan (3285 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_3285_degrees_l434_43425


namespace NUMINAMATH_CALUDE_gadget_marked_price_l434_43495

/-- The marked price of a gadget under specific conditions -/
theorem gadget_marked_price 
  (original_price : ℝ)
  (purchase_discount : ℝ)
  (desired_gain_percentage : ℝ)
  (operating_cost : ℝ)
  (selling_discount : ℝ)
  (h1 : original_price = 50)
  (h2 : purchase_discount = 0.15)
  (h3 : desired_gain_percentage = 0.4)
  (h4 : operating_cost = 5)
  (h5 : selling_discount = 0.25) :
  ∃ (marked_price : ℝ), 
    marked_price = 86 ∧ 
    marked_price * (1 - selling_discount) = 
      (original_price * (1 - purchase_discount) * (1 + desired_gain_percentage) + operating_cost) := by
  sorry


end NUMINAMATH_CALUDE_gadget_marked_price_l434_43495


namespace NUMINAMATH_CALUDE_function_parameters_l434_43438

/-- Given a function f(x) = 2sin(ωx + φ) with the specified properties, prove that ω = 2 and φ = π/3 -/
theorem function_parameters (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 →
  |φ| < π/2 →
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π) →
  f 0 = Real.sqrt 3 →
  ω = 2 ∧ φ = π/3 := by
sorry

end NUMINAMATH_CALUDE_function_parameters_l434_43438


namespace NUMINAMATH_CALUDE_equal_sharing_contribution_l434_43481

def earnings : List ℕ := [10, 30, 50, 40, 70]

theorem equal_sharing_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 30
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_equal_sharing_contribution_l434_43481


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l434_43413

/-- A sequence {a_n} with sum of first n terms S_n = p^n + q, where p ≠ 0 and p ≠ 1, 
    is geometric if and only if q = -1 -/
theorem geometric_sequence_condition (p : ℝ) (q : ℝ) (h_p_nonzero : p ≠ 0) (h_p_not_one : p ≠ 1) :
  let a : ℕ → ℝ := fun n => (p^n + q) - (p^(n-1) + q)
  let S : ℕ → ℝ := fun n => p^n + q
  (∀ n : ℕ, n ≥ 2 → a (n+1) / a n = a 2 / a 1) ↔ q = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l434_43413


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l434_43403

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l434_43403


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l434_43486

theorem complex_fraction_simplification (a b : ℝ) (ha : a = 4.91) (hb : b = 0.09) :
  (((a^2 - b^2) * (a^2 + b^(2/3) + a * b^(1/3))) / (a * b^(1/3) + a * a^(1/2) - b * b^(1/3) - (a * b^2)^(1/2))) /
  ((a^3 - b) / (a * b^(1/3) - (a^3 * b^2)^(1/6) - b^(2/3) + a * a^(1/2))) = a + b :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l434_43486


namespace NUMINAMATH_CALUDE_cubic_equation_root_l434_43480

theorem cubic_equation_root (h : ℚ) : 
  (3 : ℚ)^3 + h * 3 - 14 = 0 → h = -13/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l434_43480


namespace NUMINAMATH_CALUDE_sum_of_squares_l434_43409

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l434_43409


namespace NUMINAMATH_CALUDE_principal_is_800_l434_43449

/-- Calculates the principal amount given the final amount, interest rate, and time -/
def calculate_principal (amount : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  amount / (1 + rate * time)

/-- Theorem stating that the principal is 800 given the problem conditions -/
theorem principal_is_800 : 
  let amount : ℚ := 896
  let rate : ℚ := 5 / 100
  let time : ℚ := 12 / 5
  calculate_principal amount rate time = 800 := by sorry

end NUMINAMATH_CALUDE_principal_is_800_l434_43449


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l434_43423

theorem quadratic_root_implies_coefficient 
  (b c : ℝ) 
  (h : ∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 2 + I) : 
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l434_43423


namespace NUMINAMATH_CALUDE_g_6_eq_1_l434_43415

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the conditions on f
axiom f_1 : f 1 = 1
axiom f_add_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_add_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1 - x

-- State the theorem to be proved
theorem g_6_eq_1 : g 6 = 1 := by sorry

end NUMINAMATH_CALUDE_g_6_eq_1_l434_43415


namespace NUMINAMATH_CALUDE_meters_not_most_appropriate_for_map_distance_l434_43427

-- Define a type for units of measurement
inductive MeasurementUnit
| Meters
| Centimeters

-- Define a function to determine the most appropriate unit for map distances
def mostAppropriateUnitForMapDistance : MeasurementUnit := sorry

-- Theorem stating that meters is not the most appropriate unit
theorem meters_not_most_appropriate_for_map_distance :
  mostAppropriateUnitForMapDistance ≠ MeasurementUnit.Meters := by
  sorry

end NUMINAMATH_CALUDE_meters_not_most_appropriate_for_map_distance_l434_43427


namespace NUMINAMATH_CALUDE_locus_of_centers_l434_43478

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 25 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and
    internally tangent to C₂ satisfies the equation 4a² + 4b² - 52a - 169 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) →
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l434_43478


namespace NUMINAMATH_CALUDE_cube_preserves_inequality_l434_43467

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_inequality_l434_43467


namespace NUMINAMATH_CALUDE_opposite_teal_is_blue_l434_43429

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Teal | Violet

-- Define a cube type
structure Cube where
  faces : Fin 6 → Color
  unique_colors : ∀ i j, i ≠ j → faces i ≠ faces j

-- Define the views
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Blue ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Violet ∧ c.faces 2 = Color.Orange

-- Theorem statement
theorem opposite_teal_is_blue (c : Cube) 
  (h1 : view1 c) (h2 : view2 c) (h3 : view3 c) :
  ∃ i j, i ≠ j ∧ c.faces i = Color.Teal ∧ c.faces j = Color.Blue ∧ 
  (∀ k, k ≠ i → k ≠ j → c.faces k ≠ Color.Teal ∧ c.faces k ≠ Color.Blue) :=
sorry

end NUMINAMATH_CALUDE_opposite_teal_is_blue_l434_43429


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l434_43499

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = c and C = π/5, then B = 3π/10 -/
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l434_43499


namespace NUMINAMATH_CALUDE_airplane_passengers_l434_43485

theorem airplane_passengers (P : ℕ) 
  (h1 : P - 58 + 24 - 47 + 14 + 10 = 67) : P = 124 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l434_43485


namespace NUMINAMATH_CALUDE_point_e_satisfies_conditions_l434_43462

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: Point E(0, 0) satisfies the area ratio conditions in quadrilateral ABCD -/
theorem point_e_satisfies_conditions (A B C D E : Point) 
  (hA : A = ⟨-2, -4⟩) (hB : B = ⟨-2, 3⟩) (hC : C = ⟨4, 6⟩) (hD : D = ⟨4, -1⟩) (hE : E = ⟨0, 0⟩) :
  triangleArea E A B / triangleArea E C D = 1 / 2 ∧
  triangleArea E A D / triangleArea E B C = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_point_e_satisfies_conditions_l434_43462


namespace NUMINAMATH_CALUDE_food_bank_remaining_l434_43493

/-- Calculates the amount of food remaining in the food bank after four weeks of donations and distributions. -/
theorem food_bank_remaining (week1_donation : ℝ) (week2_factor : ℝ) (week3_increase : ℝ) (week4_decrease : ℝ)
  (week1_given_out : ℝ) (week2_given_out : ℝ) (week3_given_out : ℝ) (week4_given_out : ℝ) :
  week1_donation = 40 →
  week2_factor = 1.5 →
  week3_increase = 1.25 →
  week4_decrease = 0.9 →
  week1_given_out = 0.6 →
  week2_given_out = 0.7 →
  week3_given_out = 0.8 →
  week4_given_out = 0.5 →
  let week2_donation := week1_donation * week2_factor
  let week3_donation := week2_donation * week3_increase
  let week4_donation := week3_donation * week4_decrease
  let week1_remaining := week1_donation * (1 - week1_given_out)
  let week2_remaining := week2_donation * (1 - week2_given_out)
  let week3_remaining := week3_donation * (1 - week3_given_out)
  let week4_remaining := week4_donation * (1 - week4_given_out)
  week1_remaining + week2_remaining + week3_remaining + week4_remaining = 82.75 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_remaining_l434_43493


namespace NUMINAMATH_CALUDE_distance_of_symmetric_points_on_parabola_l434_43412

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 - x^2

-- Define the symmetry line
def symmetryLine (x y : ℝ) : Prop := x + y = 0

-- Define a point on the parabola
def pointOnParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

-- Define symmetry with respect to the line x + y = 0
def symmetricPoints (p q : ℝ × ℝ) : Prop :=
  q.1 = p.2 ∧ q.2 = p.1

-- The main theorem
theorem distance_of_symmetric_points_on_parabola (A B : ℝ × ℝ) :
  pointOnParabola A →
  pointOnParabola B →
  A ≠ B →
  symmetricPoints A B →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_of_symmetric_points_on_parabola_l434_43412


namespace NUMINAMATH_CALUDE_quadrilateral_count_l434_43474

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices required from the circumference -/
def k : ℕ := 3

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilateral_count :
  num_quadrilaterals = 455 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_count_l434_43474


namespace NUMINAMATH_CALUDE_stating_pacific_area_rounded_l434_43414

/-- The area of the Pacific Ocean in square kilometers -/
def pacific_area : ℕ := 19996800

/-- Conversion factor from square kilometers to ten thousand square kilometers -/
def ten_thousand : ℕ := 10000

/-- Rounds a natural number to the nearest multiple of ten thousand -/
def round_to_nearest_ten_thousand (n : ℕ) : ℕ :=
  (n + 5000) / 10000 * 10000

/-- 
Theorem stating that the area of the Pacific Ocean, when converted to 
ten thousand square kilometers and rounded to the nearest ten thousand, 
is equal to 2000 ten thousand square kilometers
-/
theorem pacific_area_rounded : 
  round_to_nearest_ten_thousand (pacific_area / ten_thousand) = 2000 := by
  sorry


end NUMINAMATH_CALUDE_stating_pacific_area_rounded_l434_43414


namespace NUMINAMATH_CALUDE_prime_sum_product_l434_43446

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 105 → p * q = 206 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l434_43446


namespace NUMINAMATH_CALUDE_consecutive_primes_in_sequence_l434_43435

theorem consecutive_primes_in_sequence (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ n : ℕ, n ≥ 2 → 
    ¬(Nat.Prime ((a^n - 1) / (b^n - 1)) ∧ Nat.Prime ((a^(n+1) - 1) / (b^(n+1) - 1))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_in_sequence_l434_43435


namespace NUMINAMATH_CALUDE_chess_club_officers_l434_43430

/-- The number of members in the Chess Club -/
def total_members : ℕ := 25

/-- The number of officers to be selected -/
def num_officers : ℕ := 3

/-- Function to calculate the number of ways to select officers -/
def select_officers (total : ℕ) (officers : ℕ) : ℕ :=
  let case1 := (total - 2) * (total - 3) * (total - 4)  -- Neither Alice nor Bob
  let case2 := 3 * 2 * (total - 3)  -- Both Alice and Bob
  case1 + case2

/-- Theorem stating the number of ways to select officers -/
theorem chess_club_officers :
  select_officers total_members num_officers = 10758 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_officers_l434_43430


namespace NUMINAMATH_CALUDE_digit_equality_l434_43408

theorem digit_equality (n k : ℕ) : 
  (10^(k-1) ≤ n^n ∧ n^n < 10^k) ∧ 
  (10^(n-1) ≤ k^k ∧ k^k < 10^n) ↔ 
  ((n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equality_l434_43408


namespace NUMINAMATH_CALUDE_sector_central_angle_l434_43477

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 2/5 * Real.pi) :
  (2 * area) / (r^2) = π / 5 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l434_43477


namespace NUMINAMATH_CALUDE_lemonade_production_l434_43448

/-- Given that John can prepare 15 lemonades from 3 lemons, 
    prove that he can make 90 lemonades from 18 lemons. -/
theorem lemonade_production (initial_lemons : ℕ) (initial_lemonades : ℕ) (new_lemons : ℕ) : 
  initial_lemons = 3 → initial_lemonades = 15 → new_lemons = 18 →
  (new_lemons * initial_lemonades / initial_lemons : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_production_l434_43448


namespace NUMINAMATH_CALUDE_pet_store_bird_dog_ratio_l434_43453

/-- Given a pet store with dogs, cats, birds, and fish, prove the ratio of birds to dogs. -/
theorem pet_store_bird_dog_ratio 
  (dogs : ℕ) 
  (cats : ℕ) 
  (birds : ℕ) 
  (fish : ℕ) 
  (h1 : dogs = 6) 
  (h2 : cats = dogs / 2) 
  (h3 : fish = 3 * dogs) 
  (h4 : dogs + cats + birds + fish = 39) : 
  birds / dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_dog_ratio_l434_43453


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l434_43463

theorem fraction_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (x + y) + y / (y + z) + z / (z + x) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l434_43463


namespace NUMINAMATH_CALUDE_probability_same_color_l434_43434

def num_balls : ℕ := 6
def num_colors : ℕ := 3
def balls_per_color : ℕ := 2

def same_color_combinations : ℕ := num_colors

def total_combinations : ℕ := num_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_l434_43434


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l434_43419

/-- The number of ways to choose a starting lineup for a basketball team -/
def starting_lineup_count (total_members : ℕ) (center_capable : ℕ) : ℕ :=
  center_capable * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for a specific basketball team -/
theorem basketball_lineup_count :
  starting_lineup_count 12 4 = 31680 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l434_43419


namespace NUMINAMATH_CALUDE_partnership_capital_share_l434_43426

theorem partnership_capital_share :
  let total_profit : ℚ := 2430
  let a_profit : ℚ := 810
  let a_share : ℚ := 1/3
  let b_share : ℚ := 1/4
  let d_share (c_share : ℚ) : ℚ := 1 - (a_share + b_share + c_share)
  ∀ c_share : ℚ,
    (a_share / 1 = a_profit / total_profit) →
    (a_share + b_share + c_share + d_share c_share = 1) →
    c_share = 5/24 :=
by sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l434_43426


namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l434_43410

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x + 2) + 3
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a (x + 2) + 3

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  f a (-1) = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l434_43410


namespace NUMINAMATH_CALUDE_gratuity_percentage_is_twenty_percent_l434_43455

def number_of_people : ℕ := 6
def total_bill : ℚ := 720
def average_cost_before_gratuity : ℚ := 100

theorem gratuity_percentage_is_twenty_percent :
  let total_before_gratuity : ℚ := number_of_people * average_cost_before_gratuity
  let gratuity_amount : ℚ := total_bill - total_before_gratuity
  gratuity_amount / total_before_gratuity = 1/5 := by
sorry

end NUMINAMATH_CALUDE_gratuity_percentage_is_twenty_percent_l434_43455


namespace NUMINAMATH_CALUDE_tan_sum_equals_three_l434_43458

theorem tan_sum_equals_three (α β : Real) 
  (h1 : α + β = π/3)
  (h2 : Real.sin α * Real.sin β = (Real.sqrt 3 - 3)/6) :
  Real.tan α + Real.tan β = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_three_l434_43458


namespace NUMINAMATH_CALUDE_total_books_eq_read_plus_unread_l434_43432

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 20

/-- The number of books yet to be read -/
def unread_books : ℕ := 5

/-- The number of books already read -/
def read_books : ℕ := 15

/-- Theorem stating that the total number of books is the sum of read and unread books -/
theorem total_books_eq_read_plus_unread : 
  total_books = read_books + unread_books := by
  sorry

end NUMINAMATH_CALUDE_total_books_eq_read_plus_unread_l434_43432


namespace NUMINAMATH_CALUDE_point_p_coordinates_l434_43421

/-- A point on the x-axis with distance 3 from the origin -/
structure PointP where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_3 : x^2 + y^2 = 3^2

/-- The coordinates of point P are either (-3,0) or (3,0) -/
theorem point_p_coordinates (p : PointP) : (p.x = -3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l434_43421


namespace NUMINAMATH_CALUDE_chord_length_specific_case_l434_43465

/-- The length of the chord cut by a circle on a line -/
def chord_length (a b c d e f : ℝ) : ℝ :=
  let circle := fun (x y : ℝ) => x^2 + y^2 + a*x + b*y + c
  let line := fun (x y : ℝ) => d*x + e*y + f
  -- The actual calculation of the chord length would go here
  0  -- Placeholder

theorem chord_length_specific_case :
  chord_length 0 (-2) (-1) 2 (-1) (-1) = 2 * Real.sqrt 30 / 5 := by
  sorry

#check chord_length_specific_case

end NUMINAMATH_CALUDE_chord_length_specific_case_l434_43465


namespace NUMINAMATH_CALUDE_diminished_number_divisibility_l434_43447

def smallest_number : ℕ := 1013
def diminished_number : ℕ := smallest_number - 5

def divisors : Set ℕ := {1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 96, 112, 126, 144, 168, 192, 252, 336, 504, 1008}

theorem diminished_number_divisibility :
  (∀ n ∈ divisors, diminished_number % n = 0) ∧
  (∀ m : ℕ, m > 0 → m ∉ divisors → diminished_number % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_diminished_number_divisibility_l434_43447


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l434_43440

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)

-- Define the polynomial equation
def polynomial_equation (x : ℝ) : Prop :=
  (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6

-- State the theorem
theorem sum_of_absolute_coefficients :
  (∀ x, polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅ a₆ x) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l434_43440


namespace NUMINAMATH_CALUDE_sports_equipment_purchasing_l434_43442

/-- Represents the price and quantity information for sports equipment --/
structure EquipmentInfo where
  price_a : ℕ
  price_b : ℕ
  total_budget : ℕ
  total_units : ℕ

/-- Theorem about sports equipment purchasing --/
theorem sports_equipment_purchasing (info : EquipmentInfo) 
  (h1 : 3 * info.price_a + info.price_b = 500)
  (h2 : info.price_a + 2 * info.price_b = 250)
  (h3 : info.total_budget = 2700)
  (h4 : info.total_units = 25) :
  info.price_a = 150 ∧ 
  info.price_b = 50 ∧
  (∀ m : ℕ, m * info.price_a + (info.total_units - m) * info.price_b ≤ info.total_budget → m ≤ 14) ∧
  (∀ m : ℕ, 12 ≤ m → m ≤ 14 → m * info.price_a + (info.total_units - m) * info.price_b ≥ 2450) := by
  sorry

end NUMINAMATH_CALUDE_sports_equipment_purchasing_l434_43442


namespace NUMINAMATH_CALUDE_min_intersection_length_l434_43494

def set_length (a b : ℝ) := b - a

def M (m : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ m + 7/10}
def N (n : ℝ) := {x : ℝ | n - 2/5 ≤ x ∧ x ≤ n}

theorem min_intersection_length :
  ∃ (min_length : ℝ),
    min_length = 1/10 ∧
    ∀ (m n : ℝ),
      0 ≤ m → m ≤ 3/10 →
      2/5 ≤ n → n ≤ 1 →
      ∃ (a b : ℝ),
        (∀ x, x ∈ M m ∩ N n ↔ a ≤ x ∧ x ≤ b) ∧
        set_length a b ≥ min_length :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_length_l434_43494


namespace NUMINAMATH_CALUDE_unique_solution_condition_l434_43400

theorem unique_solution_condition (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l434_43400


namespace NUMINAMATH_CALUDE_point_M_coordinates_midpoint_E_points_P₁_P₂_l434_43498

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define vertices
def A (b : ℝ) : ℝ × ℝ := (0, b)
def B (b : ℝ) : ℝ × ℝ := (0, -b)
def Q (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Theorem statements
theorem point_M_coordinates (a b : ℝ) (h : 0 < b ∧ b < a) :
  ∃ M : ℝ × ℝ, vec_add (A b) M = vec_scale (1/2) (vec_add (vec_add (A b) (Q a)) (vec_add (A b) (B b))) →
  M = (a/2, -b/2) := sorry

theorem midpoint_E (a b k₁ k₂ : ℝ) (h : k₁ * k₂ = -b^2 / a^2) :
  ∃ C D E : ℝ × ℝ, ellipse a b C.1 C.2 ∧ ellipse a b D.1 D.2 ∧
  C.2 = k₁ * C.1 + p ∧ D.2 = k₁ * D.1 + p ∧ E.2 = k₂ * E.1 ∧ E.2 = k₁ * E.1 + p →
  E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) := sorry

theorem points_P₁_P₂ (a b : ℝ) (P P₁ P₂ : ℝ × ℝ) (h₁ : a = 10 ∧ b = 5) (h₂ : P = (-8, -1)) :
  ellipse a b P₁.1 P₁.2 ∧ ellipse a b P₂.1 P₂.2 ∧
  vec_add (vec_add P P₁) (vec_add P P₂) = vec_add P (Q a) →
  (P₁ = (-6, -4) ∧ P₂ = (8, 3)) ∨ (P₁ = (8, 3) ∧ P₂ = (-6, -4)) := sorry

end NUMINAMATH_CALUDE_point_M_coordinates_midpoint_E_points_P₁_P₂_l434_43498


namespace NUMINAMATH_CALUDE_car_journey_speed_l434_43443

def car_journey (v : ℝ) : Prop :=
  let first_part_time : ℝ := 1
  let first_part_speed : ℝ := 40
  let second_part_time : ℝ := 0.5
  let third_part_time : ℝ := 2
  let total_time : ℝ := first_part_time + second_part_time + third_part_time
  let total_distance : ℝ := first_part_speed * first_part_time + v * (second_part_time + third_part_time)
  let average_speed : ℝ := 54.285714285714285
  total_distance / total_time = average_speed

theorem car_journey_speed : car_journey 60 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_speed_l434_43443


namespace NUMINAMATH_CALUDE_pencils_for_classroom_l434_43418

/-- Given a classroom with 4 children where each child receives 2 pencils,
    prove that the teacher needs to give out 8 pencils in total. -/
theorem pencils_for_classroom (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 4) (h2 : pencils_per_child = 2) : 
  num_children * pencils_per_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_for_classroom_l434_43418


namespace NUMINAMATH_CALUDE_conference_duration_l434_43475

def minutes_in_hour : ℕ := 60

def day1_hours : ℕ := 7
def day1_minutes : ℕ := 15

def day2_hours : ℕ := 8
def day2_minutes : ℕ := 45

def total_conference_minutes : ℕ := 
  (day1_hours * minutes_in_hour + day1_minutes) +
  (day2_hours * minutes_in_hour + day2_minutes)

theorem conference_duration :
  total_conference_minutes = 960 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_l434_43475


namespace NUMINAMATH_CALUDE_inequality_proof_l434_43483

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) : 
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l434_43483


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l434_43404

theorem rod_and_rope_problem (x y : ℝ) : 
  (x - y = 5 ∧ y - (1/2) * x = 5) ↔ 
  (x > y ∧ x - y = 5 ∧ y > (1/2) * x ∧ y - (1/2) * x = 5) :=
sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l434_43404


namespace NUMINAMATH_CALUDE_polynomial_equality_l434_43471

theorem polynomial_equality : 99^5 - 5*99^4 + 10*99^3 - 10*99^2 + 5*99 - 1 = 98^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l434_43471


namespace NUMINAMATH_CALUDE_projection_theorem_l434_43428

def vector_a : ℝ × ℝ := (-2, -4)

theorem projection_theorem (b : ℝ × ℝ) 
  (angle_ab : Real.cos (120 * π / 180) = -1/2)
  (magnitude_b : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5) :
  let projection := (Real.sqrt ((vector_a.1)^2 + (vector_a.2)^2)) * 
                    Real.cos (120 * π / 180)
  projection = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l434_43428


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l434_43464

def num_books : ℕ := 6
def num_people : ℕ := 3

/-- The number of ways to divide 6 books into three parts of 2 books each -/
def divide_equal_parts : ℕ := 15

/-- The number of ways to distribute 6 books to three people, each receiving 2 books -/
def distribute_equal : ℕ := 90

/-- The number of ways to distribute 6 books to three people without restrictions -/
def distribute_unrestricted : ℕ := 729

/-- The number of ways to distribute 6 books to three people, with each person receiving at least 1 book -/
def distribute_at_least_one : ℕ := 481

theorem book_distribution_theorem :
  divide_equal_parts = 15 ∧
  distribute_equal = 90 ∧
  distribute_unrestricted = 729 ∧
  distribute_at_least_one = 481 :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l434_43464


namespace NUMINAMATH_CALUDE_boat_downstream_time_l434_43433

theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (current_rate : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 18) 
  (h2 : current_rate = 4) 
  (h3 : distance = 5.133333333333334) : 
  (distance / (boat_speed + current_rate)) * 60 = 14 := by
sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l434_43433


namespace NUMINAMATH_CALUDE_largest_positive_integer_for_binary_op_l434_43445

def binary_op (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer_for_binary_op :
  ∀ n : ℕ+, n > 1 → binary_op n.val ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_for_binary_op_l434_43445


namespace NUMINAMATH_CALUDE_line_circle_intersection_l434_43431

/-- A line y = kx + 3 intersects a circle (x - 3)^2 + (y - 2)^2 = 4 at two points M and N. 
    If the distance between M and N is at least 2, then k is outside the interval (3 - 2√2, 3 + 2√2). -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧
    (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 4) →
  k < 3 - 2 * Real.sqrt 2 ∨ k > 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l434_43431


namespace NUMINAMATH_CALUDE_v_2002_equals_4_l434_43490

def g : ℕ → ℕ
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | 5 => 5
  | _ => 0  -- default case for completeness

def v : ℕ → ℕ
  | 0 => 3
  | n + 1 => g (v n)

theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_4_l434_43490
