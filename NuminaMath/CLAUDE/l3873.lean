import Mathlib

namespace division_problem_l3873_387329

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := by
  sorry

end division_problem_l3873_387329


namespace range_of_a_l3873_387372

theorem range_of_a (x a : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → (x - a) * (x - a - 4) > 0) ∧ 
  (∃ x, (x - a) * (x - a - 4) > 0 ∧ (x < -2 ∨ x > 1)) →
  a < -6 ∨ a > 1 := by
sorry

end range_of_a_l3873_387372


namespace necessary_not_sufficient_condition_l3873_387361

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end necessary_not_sufficient_condition_l3873_387361


namespace computer_table_cost_price_l3873_387307

/-- Proves that the cost price of a computer table is 6625 when the selling price is 8215 with a 24% markup -/
theorem computer_table_cost_price (selling_price : ℕ) (markup_percentage : ℕ) (cost_price : ℕ) :
  selling_price = 8215 →
  markup_percentage = 24 →
  selling_price = cost_price + (cost_price * markup_percentage) / 100 →
  cost_price = 6625 := by
  sorry

end computer_table_cost_price_l3873_387307


namespace quadrilateral_inscribed_circle_l3873_387319

-- Define the types for points and circles
variable (Point : Type) (Circle : Type)

-- Define the necessary geometric predicates
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (on_segment : Point → Point → Point → Prop)
variable (intersection : Point → Point → Point → Point → Point → Prop)
variable (has_inscribed_circle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem quadrilateral_inscribed_circle 
  (A B C D E F G H P : Point) :
  is_convex_quadrilateral A B C D →
  on_segment A B E →
  on_segment B C F →
  on_segment C D G →
  on_segment D A H →
  intersection E G F H P →
  has_inscribed_circle H A E P →
  has_inscribed_circle E B F P →
  has_inscribed_circle F C G P →
  has_inscribed_circle G D H P →
  has_inscribed_circle A B C D :=
by sorry

end quadrilateral_inscribed_circle_l3873_387319


namespace range_of_a_l3873_387379

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f (-a*x + Real.log x + 1) + f (a*x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / Real.exp 1) ((2 + Real.log 3) / 3) :=
sorry

end range_of_a_l3873_387379


namespace given_circles_are_externally_tangent_l3873_387366

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- Definition of the given circles --/
def givenCircles : TwoCircles where
  c1 := fun (x, y) ↦ x^2 + y^2 - 4*x - 6*y + 9 = 0
  c2 := fun (x, y) ↦ x^2 + y^2 + 12*x + 6*y - 19 = 0

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii --/
def areExternallyTangent (circles : TwoCircles) : Prop :=
  ∃ (x1 y1 x2 y2 r1 r2 : ℝ),
    (∀ (x y : ℝ), circles.c1 (x, y) ↔ (x - x1)^2 + (y - y1)^2 = r1^2) ∧
    (∀ (x y : ℝ), circles.c2 (x, y) ↔ (x - x2)^2 + (y - y2)^2 = r2^2) ∧
    (x2 - x1)^2 + (y2 - y1)^2 = (r1 + r2)^2

/-- Theorem stating that the given circles are externally tangent --/
theorem given_circles_are_externally_tangent :
  areExternallyTangent givenCircles := by sorry

end given_circles_are_externally_tangent_l3873_387366


namespace kelly_apples_l3873_387383

def initial_apples : ℕ := 56
def picked_apples : ℝ := 105.0
def total_apples : ℕ := 161

theorem kelly_apples : 
  initial_apples + picked_apples = total_apples :=
by sorry

end kelly_apples_l3873_387383


namespace money_division_l3873_387360

theorem money_division (a b c : ℚ) : 
  a = (1/2) * b ∧ b = (1/2) * c ∧ c = 232 → a + b + c = 406 := by
  sorry

end money_division_l3873_387360


namespace pool_width_is_40_l3873_387335

/-- Represents a rectangular pool with given length and width -/
structure Pool where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular pool -/
def Pool.perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Represents the speeds of Ruth and Sarah -/
structure Speeds where
  ruth : ℝ
  sarah : ℝ

theorem pool_width_is_40 (p : Pool) (s : Speeds) : p.width = 40 :=
  by
  have h1 : p.length = 50 := by sorry
  have h2 : s.ruth = 3 * s.sarah := by sorry
  have h3 : 6 * p.length = 5 * p.perimeter := by sorry
  sorry

end pool_width_is_40_l3873_387335


namespace book_selection_ways_l3873_387387

def num_books : ℕ := 5
def num_students : ℕ := 2

theorem book_selection_ways :
  (num_books ^ num_students : ℕ) = 25 := by
  sorry

end book_selection_ways_l3873_387387


namespace sum_of_opposite_sign_integers_l3873_387312

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 6) → (abs b = 4) → (a * b < 0) → (a + b = 2 ∨ a + b = -2) := by
sorry

end sum_of_opposite_sign_integers_l3873_387312


namespace log2_derivative_l3873_387388

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end log2_derivative_l3873_387388


namespace tyler_initial_money_l3873_387331

def scissors_cost : ℕ := 8 * 5
def erasers_cost : ℕ := 10 * 4
def remaining_money : ℕ := 20

theorem tyler_initial_money :
  scissors_cost + erasers_cost + remaining_money = 100 :=
by sorry

end tyler_initial_money_l3873_387331


namespace non_egg_laying_hens_l3873_387350

/-- Proves that the number of non-egg-laying hens is 20 given the total number of chickens,
    number of roosters, and number of egg-laying hens. -/
theorem non_egg_laying_hens (total_chickens roosters egg_laying_hens : ℕ) : 
  total_chickens = 325 →
  roosters = 28 →
  egg_laying_hens = 277 →
  total_chickens - roosters - egg_laying_hens = 20 := by
sorry

end non_egg_laying_hens_l3873_387350


namespace rice_cake_slices_l3873_387375

theorem rice_cake_slices (num_cakes : ℕ) (cake_length : ℝ) (overlap : ℝ) (num_slices : ℕ) :
  num_cakes = 5 →
  cake_length = 2.7 →
  overlap = 0.3 →
  num_slices = 6 →
  (num_cakes * cake_length - (num_cakes - 1) * overlap) / num_slices = 2.05 := by
  sorry

end rice_cake_slices_l3873_387375


namespace play_recording_distribution_l3873_387303

theorem play_recording_distribution (play_duration : ℕ) (disc_capacity : ℕ) 
  (h1 : play_duration = 385)
  (h2 : disc_capacity = 75) : 
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * disc_capacity ≥ play_duration ∧
    (num_discs - 1) * disc_capacity < play_duration ∧
    play_duration / num_discs = 64 := by
  sorry

end play_recording_distribution_l3873_387303


namespace stating_b_joined_after_five_months_l3873_387316

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment -/
def aInvestment : ℕ := 3500

/-- Represents B's investment -/
def bInvestment : ℕ := 9000

/-- Represents the profit ratio for A -/
def aProfitRatio : ℕ := 2

/-- Represents the profit ratio for B -/
def bProfitRatio : ℕ := 3

/-- 
Theorem stating that B joined 5 months after A started the business,
given the conditions of the problem.
-/
theorem b_joined_after_five_months :
  ∀ (x : ℕ),
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = aProfitRatio / bProfitRatio →
  x = 5 := by
  sorry


end stating_b_joined_after_five_months_l3873_387316


namespace order_of_exponentials_l3873_387382

theorem order_of_exponentials : 
  let a : ℝ := (2 : ℝ) ^ (1/5 : ℝ)
  let b : ℝ := (2/5 : ℝ) ^ (1/5 : ℝ)
  let c : ℝ := (2/5 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end order_of_exponentials_l3873_387382


namespace even_number_power_of_two_l3873_387359

theorem even_number_power_of_two (A : ℕ) :
  A % 2 = 0 →
  (∀ P : ℕ, Nat.Prime P → P ∣ A → (P - 1) ∣ (A - 1)) →
  ∃ k : ℕ, A = 2^k :=
sorry

end even_number_power_of_two_l3873_387359


namespace elvin_first_month_bill_l3873_387371

/-- Represents Elvin's monthly telephone bill structure -/
structure PhoneBill where
  fixed_charge : ℝ
  call_charge : ℝ

/-- Calculates the total bill given a PhoneBill -/
def total_bill (bill : PhoneBill) : ℝ :=
  bill.fixed_charge + bill.call_charge

theorem elvin_first_month_bill :
  ∀ (bill1 bill2 : PhoneBill),
    total_bill bill1 = 52 →
    total_bill bill2 = 76 →
    bill2.call_charge = 2 * bill1.call_charge →
    bill1.fixed_charge = bill2.fixed_charge →
    total_bill bill1 = 52 := by
  sorry

end elvin_first_month_bill_l3873_387371


namespace prob_odd_fair_die_l3873_387391

def die_outcomes : Finset Nat := {1, 2, 3, 4, 5, 6}
def odd_outcomes : Finset Nat := {1, 3, 5}

theorem prob_odd_fair_die :
  (Finset.card odd_outcomes : ℚ) / (Finset.card die_outcomes : ℚ) = 1 / 2 :=
by sorry

end prob_odd_fair_die_l3873_387391


namespace number_of_elements_in_set_l3873_387385

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (final_avg : ℚ) : 
  initial_avg = 17 →
  incorrect_num = 26 →
  correct_num = 56 →
  final_avg = 20 →
  (∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg + (correct_num - incorrect_num) ∧ n = 10) :=
by sorry

end number_of_elements_in_set_l3873_387385


namespace at_least_one_less_than_two_l3873_387323

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end at_least_one_less_than_two_l3873_387323


namespace distance_between_trees_l3873_387318

/-- Proves that in a yard of given length with a given number of trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is as calculated. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 250)
  (h2 : num_trees = 51)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 5 :=
sorry

end distance_between_trees_l3873_387318


namespace smallest_number_proof_l3873_387392

/-- The smallest natural number divisible by 21 with exactly 105 distinct divisors -/
def smallest_number_with_properties : ℕ := 254016

/-- The number of distinct divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  (smallest_number_with_properties % 21 = 0) ∧
  (num_divisors smallest_number_with_properties = 105) ∧
  (∀ m : ℕ, m < smallest_number_with_properties →
    ¬(m % 21 = 0 ∧ num_divisors m = 105)) :=
by sorry

end smallest_number_proof_l3873_387392


namespace regular_polygon_sides_l3873_387367

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l3873_387367


namespace parabola_equation_l3873_387333

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ      -- y-coordinate of the focus

/-- The standard equation of a parabola. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = -4 * p.directrix * y) ↔ (y = p.directrix ∨ (x^2 + (y - p.focus)^2 = (y - p.directrix)^2))

/-- Theorem: For a parabola with directrix y = 4, its standard equation is x² = -16y. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = 4) : 
  standardEquation p ↔ ∀ x y : ℝ, x^2 = -16*y ↔ (y = 4 ∨ (x^2 + (y - p.focus)^2 = (y - 4)^2)) :=
sorry

end parabola_equation_l3873_387333


namespace parallel_line_k_value_l3873_387337

/-- Given a line passing through points (5, -3) and (k, 20) that is parallel to the line 3x - 2y = 12, 
    prove that k = 61/3 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + b → (x = 5 ∧ y = -3) ∨ (x = k ∧ y = 20)) ∧
                (∀ x y : ℚ, 3 * x - 2 * y = 12 → y = m * x - 6)) → 
  k = 61 / 3 := by
sorry

end parallel_line_k_value_l3873_387337


namespace all_functions_are_zero_l3873_387346

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x y : ℕ, f (x * y) = f x + f y) ∧
  (f 30 = 0) ∧
  (∀ x : ℕ, x % 10 = 7 → f x = 0)

theorem all_functions_are_zero (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, f n = 0 := by
  sorry

end all_functions_are_zero_l3873_387346


namespace age_difference_l3873_387357

theorem age_difference (a b c d : ℤ) 
  (total_ab_cd : a + b = c + d + 20)
  (total_bd_ac : b + d = a + c + 10) :
  d = a - 5 := by
  sorry

end age_difference_l3873_387357


namespace arctan_sum_equals_pi_over_six_l3873_387322

theorem arctan_sum_equals_pi_over_six (b : ℝ) :
  (4/3 : ℝ) * (b + 1) = 3/2 →
  Real.arctan (1/3) + Real.arctan b = π/6 := by
  sorry

end arctan_sum_equals_pi_over_six_l3873_387322


namespace derivative_zero_neither_necessary_nor_sufficient_l3873_387373

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (a - ε) (a + ε), f x ≤ f a ∨ f x ≥ f a

-- Define the statement to be proven
theorem derivative_zero_neither_necessary_nor_sufficient :
  ¬(∀ f : ℝ → ℝ, ∀ a : ℝ, (has_extremum f a ↔ HasDerivAt f 0 a)) :=
sorry

end derivative_zero_neither_necessary_nor_sufficient_l3873_387373


namespace diagonals_100_sided_polygon_l3873_387354

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 100 sides is 4850 -/
theorem diagonals_100_sided_polygon : num_diagonals 100 = 4850 := by
  sorry

end diagonals_100_sided_polygon_l3873_387354


namespace probability_of_y_selection_l3873_387352

theorem probability_of_y_selection 
  (p_x : ℝ) 
  (p_both : ℝ) 
  (h1 : p_x = 1 / 7)
  (h2 : p_both = 0.05714285714285714) :
  p_both / p_x = 0.4 := by
sorry

end probability_of_y_selection_l3873_387352


namespace correct_proposition_l3873_387365

-- Define proposition p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem statement
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end correct_proposition_l3873_387365


namespace distinct_roots_imply_distinct_roots_l3873_387334

theorem distinct_roots_imply_distinct_roots (p q : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (h1 : (p^2 - 4*q) > 0) 
  (h2 : (q^2 - 4*p) > 0) : 
  ((p + q)^2 - 8*(p + q)) > 0 := by
sorry


end distinct_roots_imply_distinct_roots_l3873_387334


namespace next_year_with_digit_sum_five_l3873_387396

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem next_year_with_digit_sum_five : 
  ∀ y : ℕ, y > 2021 ∧ y < 2030 → sum_of_digits y ≠ 5 ∧ sum_of_digits 2030 = 5 :=
by sorry

end next_year_with_digit_sum_five_l3873_387396


namespace unique_a_l3873_387363

theorem unique_a : ∃! a : ℝ, (∃ m : ℤ, a + 2/3 = m) ∧ (∃ n : ℤ, 1/a - 3/4 = n) := by
  sorry

end unique_a_l3873_387363


namespace markup_is_ten_l3873_387378

/-- Calculates the markup given shop price, tax rate, and profit -/
def calculate_markup (shop_price : ℝ) (tax_rate : ℝ) (profit : ℝ) : ℝ :=
  shop_price - (shop_price * (1 - tax_rate) - profit)

theorem markup_is_ten :
  let shop_price : ℝ := 90
  let tax_rate : ℝ := 0.1
  let profit : ℝ := 1
  calculate_markup shop_price tax_rate profit = 10 := by
sorry

end markup_is_ten_l3873_387378


namespace circle_equation_l3873_387364

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point :=
  ⟨1, 0⟩

/-- The line passing through the focus with slope angle 30° -/
def Line : Set Point :=
  {p : Point | p.y = (Real.sqrt 3 / 3) * (p.x - 1)}

/-- Intersection points of the parabola and the line -/
def intersectionPoints : Set Point :=
  Parabola ∩ Line

/-- The circle with AB as diameter -/
def Circle (A B : Point) : Set Point :=
  {p : Point | (p.x - (A.x + B.x) / 2)^2 + (p.y - (A.y + B.y) / 2)^2 = ((A.x - B.x)^2 + (A.y - B.y)^2) / 4}

theorem circle_equation (A B : Point) 
  (hA : A ∈ intersectionPoints) (hB : B ∈ intersectionPoints) (hAB : A ≠ B) :
  Circle A B = {p : Point | (p.x - 7)^2 + (p.y - 2 * Real.sqrt 3)^2 = 64} :=
sorry

end circle_equation_l3873_387364


namespace school_sections_l3873_387340

/-- The number of sections formed when dividing boys and girls into equal groups -/
def total_sections (num_boys num_girls : ℕ) : ℕ :=
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls))

/-- Theorem stating that for 408 boys and 192 girls, the total number of sections is 25 -/
theorem school_sections : total_sections 408 192 = 25 := by
  sorry

end school_sections_l3873_387340


namespace z_coordinate_at_x_7_l3873_387395

/-- A line in 3D space passing through two points -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Get the z-coordinate of a point on the line given its x-coordinate -/
def get_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

theorem z_coordinate_at_x_7 (line : Line3D) 
  (h1 : line.point1 = (1, 4, 3)) 
  (h2 : line.point2 = (4, 3, 0)) : 
  get_z_coordinate line 7 = -3 := by
  sorry

end z_coordinate_at_x_7_l3873_387395


namespace base_13_conversion_l3873_387332

-- Define a function to convert a base 10 number to base 13
def toBase13 (n : ℕ) : String :=
  sorry

-- Define a function to convert a base 13 string to base 10
def fromBase13 (s : String) : ℕ :=
  sorry

-- Theorem statement
theorem base_13_conversion :
  toBase13 136 = "A6" ∧ fromBase13 "A6" = 136 :=
sorry

end base_13_conversion_l3873_387332


namespace association_confidence_level_l3873_387351

-- Define the χ² value
def chi_squared : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level with 1 degree of freedom
def critical_value : ℝ := 6.635

-- Define the confidence level we want to prove
def target_confidence_level : ℝ := 99

-- Theorem statement
theorem association_confidence_level :
  chi_squared > critical_value →
  (∃ (confidence_level : ℝ), confidence_level ≥ target_confidence_level) :=
sorry

end association_confidence_level_l3873_387351


namespace evaluate_expression_l3873_387325

theorem evaluate_expression : (49^2 - 35^2) + (15^2 - 9^2) = 1320 := by
  sorry

end evaluate_expression_l3873_387325


namespace quadratic_equation_result_l3873_387314

theorem quadratic_equation_result (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : 
  (8 * y - 4)^2 = 202 := by
  sorry

end quadratic_equation_result_l3873_387314


namespace equation_solution_unique_l3873_387339

theorem equation_solution_unique :
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x :=
by
  -- Proof goes here
  sorry

end equation_solution_unique_l3873_387339


namespace total_cost_is_correct_l3873_387397

-- Define the prices and quantities
def pencil_price : ℚ := 0.5
def folder_price : ℚ := 0.9
def notebook_price : ℚ := 1.2
def stapler_price : ℚ := 2.5

def pencil_quantity : ℕ := 24
def folder_quantity : ℕ := 20
def notebook_quantity : ℕ := 15
def stapler_quantity : ℕ := 10

-- Define discount rates
def pencil_discount_rate : ℚ := 0.1
def folder_discount_rate : ℚ := 0.15

-- Define discount conditions
def pencil_discount_threshold : ℕ := 15
def folder_discount_threshold : ℕ := 10

-- Define notebook offer
def notebook_offer : ℕ := 3  -- buy 2 get 1 free

-- Define the total cost function
def total_cost : ℚ :=
  let pencil_cost := pencil_price * pencil_quantity * (1 - pencil_discount_rate)
  let folder_cost := folder_price * folder_quantity * (1 - folder_discount_rate)
  let notebook_cost := notebook_price * (notebook_quantity - notebook_quantity / notebook_offer)
  let stapler_cost := stapler_price * stapler_quantity
  pencil_cost + folder_cost + notebook_cost + stapler_cost

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 63.1 := by
  sorry

end total_cost_is_correct_l3873_387397


namespace polynomial_equation_l3873_387358

variables (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 - x + 5

def g (x : ℝ) : ℝ := -x^4 + 7*x^2 + x - 6

theorem polynomial_equation :
  f x + g x = 4*x^2 + x - 1 := by sorry

end polynomial_equation_l3873_387358


namespace max_value_of_a_l3873_387362

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_six : a^2 + b^2 + c^2 = 6) : 
  ∀ x : ℝ, x ≤ 2 ∧ (∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 6 ∧ a₀ = 2) :=
by sorry

end max_value_of_a_l3873_387362


namespace all_lines_pass_through_fixed_point_l3873_387309

/-- The line equation passing through a fixed point for all real a -/
def line_equation (a x y : ℝ) : Prop :=
  (a + 1) * x + y - 2 - a = 0

/-- The fixed point that all lines pass through -/
def fixed_point : ℝ × ℝ := (1, 1)

/-- Theorem: All lines in the family pass through the fixed point (1, 1) -/
theorem all_lines_pass_through_fixed_point :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) :=
by
  sorry

end all_lines_pass_through_fixed_point_l3873_387309


namespace ln_plus_x_eq_three_solution_exists_in_two_three_l3873_387348

open Real

theorem ln_plus_x_eq_three_solution_exists_in_two_three :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x = 3 := by
  sorry

end ln_plus_x_eq_three_solution_exists_in_two_three_l3873_387348


namespace divisible_by_six_count_and_percentage_l3873_387317

theorem divisible_by_six_count_and_percentage :
  let n : ℕ := 120
  let divisible_count : ℕ := (n / 6 : ℕ)
  divisible_count = 20 ∧ 
  (divisible_count : ℚ) / n * 100 = 50 / 3 := by
  sorry

end divisible_by_six_count_and_percentage_l3873_387317


namespace emilys_final_score_l3873_387304

/-- A trivia game with 5 rounds and specific scoring rules -/
def triviaGame (round1 round2 round3 round4Base round5Base lastRoundLoss : ℕ) 
               (round4Multiplier round5Multiplier : ℕ) : ℕ :=
  round1 + round2 + round3 + 
  (round4Base * round4Multiplier) + 
  (round5Base * round5Multiplier) - 
  lastRoundLoss

/-- The final score of Emily's trivia game -/
theorem emilys_final_score : 
  triviaGame 16 33 21 10 4 48 2 3 = 54 := by
  sorry

end emilys_final_score_l3873_387304


namespace max_citizens_for_minister_l3873_387345

theorem max_citizens_for_minister (n : ℕ) : 
  (∀ m : ℕ, m > n → Nat.choose m 4 ≥ Nat.choose m 2) ↔ n = 5 :=
by sorry

end max_citizens_for_minister_l3873_387345


namespace calculator_presses_to_exceed_250_l3873_387341

def calculator_function (x : ℕ) : ℕ := x^2 + 3

def iterate_calculator (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => calculator_function (iterate_calculator n m)

theorem calculator_presses_to_exceed_250 :
  ∃ n : ℕ, n > 0 ∧ iterate_calculator n 3 > 250 ∧ ∀ m : ℕ, m < n → iterate_calculator n m ≤ 250 :=
by sorry

end calculator_presses_to_exceed_250_l3873_387341


namespace weight_of_b_l3873_387377

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 40 := by
  sorry

end weight_of_b_l3873_387377


namespace arithmetic_mean_difference_l3873_387302

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := by
sorry

end arithmetic_mean_difference_l3873_387302


namespace quadratic_points_ordering_l3873_387374

theorem quadratic_points_ordering (m : ℝ) (y₁ y₂ y₃ : ℝ) :
  ((-1)^2 + 2*(-1) + m = y₁) →
  (3^2 + 2*3 + m = y₂) →
  ((1/2)^2 + 2*(1/2) + m = y₃) →
  (y₂ > y₃ ∧ y₃ > y₁) :=
by sorry

end quadratic_points_ordering_l3873_387374


namespace subset_implies_a_squared_equals_two_l3873_387398

/-- Given sets A and B, where A = {0, 2, 3} and B = {2, a² + 1}, and B is a subset of A,
    prove that a² = 2, where a is a real number. -/
theorem subset_implies_a_squared_equals_two (a : ℝ) : 
  let A : Set ℝ := {0, 2, 3}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a^2 = 2 := by
  sorry

end subset_implies_a_squared_equals_two_l3873_387398


namespace z_value_z_value_proof_l3873_387320

theorem z_value : ℝ → ℝ → ℝ → Prop :=
  fun x y z =>
    x = 40 * (1 + 0.2) →
    y = x * (1 - 0.35) →
    z = (x + y) / 2 →
    z = 39.6

-- Proof
theorem z_value_proof : ∃ x y z : ℝ, z_value x y z := by
  sorry

end z_value_z_value_proof_l3873_387320


namespace max_value_constraint_l3873_387376

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = 5 ∧ ∀ (a b : ℝ), a^2 + b^2 + a*b = 1 → 3*a - 2*b ≤ M :=
by sorry

end max_value_constraint_l3873_387376


namespace job_completion_time_l3873_387301

/-- If m men can do a job in d days, and n men can do a different job in k days,
    then m+n men can do both jobs in (m * d + n * k) / (m + n) days. -/
theorem job_completion_time
  (m n d k : ℕ) (hm : m > 0) (hn : n > 0) (hd : d > 0) (hk : k > 0) :
  let total_time := (m * d + n * k) / (m + n)
  ∃ (time : ℚ), time = total_time ∧ time > 0 := by
  sorry

end job_completion_time_l3873_387301


namespace base12_divisibility_rule_l3873_387353

/-- 
Represents a number in base-12 as a list of digits, 
where each digit is between 0 and 11 (inclusive).
--/
def Base12Number := List Nat

/-- Converts a Base12Number to its decimal representation. --/
def toDecimal (n : Base12Number) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (12 ^ i)) 0

/-- Calculates the sum of digits in a Base12Number. --/
def digitSum (n : Base12Number) : Nat :=
  n.sum

theorem base12_divisibility_rule (n : Base12Number) :
  11 ∣ digitSum n → 11 ∣ toDecimal n := by sorry

end base12_divisibility_rule_l3873_387353


namespace system_solution_l3873_387343

theorem system_solution : ∃! (x y : ℝ), 
  (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
  (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
  x = (1 : ℝ) / 2 ∧ y = -(3 : ℝ) / 4 := by
  sorry

end system_solution_l3873_387343


namespace remainder_of_2743_base12_div_9_l3873_387326

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (n : ℕ) : ℕ :=
  let d0 := n % 12
  let d1 := (n / 12) % 12
  let d2 := (n / 144) % 12
  let d3 := n / 1728
  d3 * 1728 + d2 * 144 + d1 * 12 + d0

/-- The base-12 number 2743 --/
def n : ℕ := 2743

theorem remainder_of_2743_base12_div_9 :
  (base12ToBase10 n) % 9 = 0 := by
  sorry

end remainder_of_2743_base12_div_9_l3873_387326


namespace no_solution_equation1_unique_solution_equation2_l3873_387327

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 - x) / (x - 2) + 1 / (2 - x) = 1
def equation2 (x : ℝ) : Prop := 3 / (x^2 - 4) + 2 / (x + 2) = 1 / (x - 2)

-- Theorem for equation 1
theorem no_solution_equation1 : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem unique_solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = 3 := by sorry

end no_solution_equation1_unique_solution_equation2_l3873_387327


namespace unique_triple_l3873_387355

theorem unique_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 1 ∧ (x, y, z) = (2, 2, 0) := by
  sorry

end unique_triple_l3873_387355


namespace island_inhabitants_l3873_387380

theorem island_inhabitants (total : Nat) (blue_eyed : Nat) (brown_eyed : Nat) : 
  total = 100 →
  blue_eyed + brown_eyed = total →
  (blue_eyed * brown_eyed * 2 > (total * (total - 1)) / 2) →
  (∀ (x : Nat), x ≤ blue_eyed → x ≤ brown_eyed → x * (total - x) ≤ blue_eyed * brown_eyed) →
  46 ≤ brown_eyed ∧ brown_eyed ≤ 54 := by
  sorry

end island_inhabitants_l3873_387380


namespace ball_distribution_probability_ratio_l3873_387381

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let num_bins : ℕ := 5
  let prob_6_7_4_4_4 := (Nat.choose num_bins 2 * Nat.choose total_balls 6 * Nat.choose 19 7 * 
                         Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
                        (num_bins ^ total_balls : ℚ)
  let prob_5_5_5_5_5 := (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * 
                         Nat.choose 10 5 * Nat.choose 5 5) / 
                        (num_bins ^ total_balls : ℚ)
  prob_6_7_4_4_4 / prob_5_5_5_5_5 = 
    (10 * Nat.choose total_balls 6 * Nat.choose 19 7 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
    (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * Nat.choose 10 5 * Nat.choose 5 5)
  := by sorry

end ball_distribution_probability_ratio_l3873_387381


namespace sequence_value_proof_l3873_387356

def sequence_sum (n : ℕ) : ℤ := n^2 - 9*n

def sequence_term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n-1)

theorem sequence_value_proof (k : ℕ) (h : 5 < sequence_term k ∧ sequence_term k < 8) :
  sequence_term k = 6 := by sorry

end sequence_value_proof_l3873_387356


namespace gcd_f_x_eq_one_l3873_387311

def f (x : ℤ) : ℤ := (3*x+4)*(8*x+5)*(15*x+11)*(x+14)

theorem gcd_f_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 54321 * k) :
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 1 := by
sorry

end gcd_f_x_eq_one_l3873_387311


namespace q_prime_div_p_prime_eq_550_l3873_387328

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 12

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p' : ℚ := 12 / (total_slips.choose drawn_slips)

/-- The probability of drawing three slips with one number and two with another -/
def q' : ℚ := (6600 : ℚ) / (total_slips.choose drawn_slips)

/-- The main theorem stating the ratio of q' to p' -/
theorem q_prime_div_p_prime_eq_550 : q' / p' = 550 := by sorry

end q_prime_div_p_prime_eq_550_l3873_387328


namespace circle_no_intersection_probability_l3873_387321

/-- A rectangle with width 15 and height 36 -/
structure Rectangle where
  width : ℝ := 15
  height : ℝ := 36

/-- A circle with radius 1 -/
structure Circle where
  radius : ℝ := 1

/-- The probability that a circle doesn't intersect the diagonal of a rectangle -/
def probability_no_intersection (r : Rectangle) (c : Circle) : ℚ :=
  375 / 442

/-- Theorem stating the probability of no intersection -/
theorem circle_no_intersection_probability (r : Rectangle) (c : Circle) :
  probability_no_intersection r c = 375 / 442 := by
  sorry

#check circle_no_intersection_probability

end circle_no_intersection_probability_l3873_387321


namespace scientific_notation_1742000_l3873_387370

theorem scientific_notation_1742000 :
  ∃ (a : ℝ) (n : ℤ), 1742000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  use 1.742, 6
  sorry

end scientific_notation_1742000_l3873_387370


namespace multiple_in_denominator_l3873_387347

theorem multiple_in_denominator (a b k : ℚ) : 
  a / b = 4 / 1 →
  (a - 3 * b) / (k * (a - b)) = 1 / 7 →
  k = 7 / 3 := by sorry

end multiple_in_denominator_l3873_387347


namespace opposite_of_negative_fraction_l3873_387394

theorem opposite_of_negative_fraction :
  -(-(1 / 2024)) = 1 / 2024 := by
  sorry

end opposite_of_negative_fraction_l3873_387394


namespace chess_tournament_players_l3873_387315

theorem chess_tournament_players :
  ∀ (num_girls : ℕ) (total_points : ℕ),
    (num_girls > 0) →
    (total_points = 2 * num_girls * (6 * num_girls - 1)) →
    (2 * num_girls * (6 * num_girls - 1) = (num_girls^2 + 9*num_girls) + 2*(5*num_girls)*(5*num_girls - 1)) →
    (num_girls + 5*num_girls = 6) :=
by
  sorry

end chess_tournament_players_l3873_387315


namespace junhyun_travel_distance_l3873_387300

/-- The distance Junhyun traveled by bus in kilometers -/
def bus_distance : ℝ := 2.6

/-- The distance Junhyun traveled by subway in kilometers -/
def subway_distance : ℝ := 5.98

/-- The total distance Junhyun traveled using public transportation -/
def total_distance : ℝ := bus_distance + subway_distance

/-- Theorem stating that the total distance Junhyun traveled is 8.58 km -/
theorem junhyun_travel_distance : total_distance = 8.58 := by sorry

end junhyun_travel_distance_l3873_387300


namespace simplify_fourth_root_exponent_sum_l3873_387393

theorem simplify_fourth_root_exponent_sum (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ) (n m : ℕ), (48 * a^5 * b^8 * c^14)^(1/4) = k * b^n * c^m ∧ n + m = 5 :=
sorry

end simplify_fourth_root_exponent_sum_l3873_387393


namespace pear_count_theorem_l3873_387313

/-- Represents the types of fruits on the table -/
inductive Fruit
  | Apple
  | Pear
  | Orange

/-- Represents the state of the table -/
structure TableState where
  apples : Nat
  pears : Nat
  oranges : Nat

/-- Defines the order in which fruits are taken -/
def nextFruit : Fruit → Fruit
  | Fruit.Apple => Fruit.Pear
  | Fruit.Pear => Fruit.Orange
  | Fruit.Orange => Fruit.Apple

/-- Determines if a fruit can be taken from the table -/
def canTakeFruit (state : TableState) (fruit : Fruit) : Bool :=
  match fruit with
  | Fruit.Apple => state.apples > 0
  | Fruit.Pear => state.pears > 0
  | Fruit.Orange => state.oranges > 0

/-- Takes a fruit from the table -/
def takeFruit (state : TableState) (fruit : Fruit) : TableState :=
  match fruit with
  | Fruit.Apple => { state with apples := state.apples - 1 }
  | Fruit.Pear => { state with pears := state.pears - 1 }
  | Fruit.Orange => { state with oranges := state.oranges - 1 }

/-- Checks if the table is empty -/
def isTableEmpty (state : TableState) : Bool :=
  state.apples = 0 && state.pears = 0 && state.oranges = 0

/-- Main theorem: The number of pears must be either 99 or 100 -/
theorem pear_count_theorem (initialPears : Nat) :
  let initialState : TableState := { apples := 100, pears := initialPears, oranges := 99 }
  (∃ (finalState : TableState), 
    isTableEmpty finalState ∧
    (∀ fruit : Fruit, canTakeFruit initialState fruit →
      ∃ nextState : TableState, 
        nextState = takeFruit initialState fruit ∧
        (isTableEmpty nextState ∨ 
          canTakeFruit nextState (nextFruit fruit)))) →
  initialPears = 99 ∨ initialPears = 100 := by
  sorry


end pear_count_theorem_l3873_387313


namespace average_transformation_l3873_387384

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end average_transformation_l3873_387384


namespace set_operation_result_l3873_387310

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 2, 3}
def N : Finset Nat := {3, 4, 5}

theorem set_operation_result : 
  (U \ M) ∩ N = {4, 5} := by sorry

end set_operation_result_l3873_387310


namespace complex_magnitude_problem_l3873_387336

variable (w z : ℂ)

theorem complex_magnitude_problem (h1 : w * z = 24 - 10 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (13 * Real.sqrt 34) / 17 := by
  sorry

end complex_magnitude_problem_l3873_387336


namespace infinitely_many_satisfying_functions_l3873_387368

/-- A function that satisfies the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = x^2) ∧ (Set.range f = Set.Icc 1 4)

/-- There exist infinitely many functions satisfying the given conditions -/
theorem infinitely_many_satisfying_functions :
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ f ∈ S, satisfying_function f :=
sorry

end infinitely_many_satisfying_functions_l3873_387368


namespace ellipse_parabola_intersection_l3873_387349

/-- An ellipse passing through (1,0) with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eq : 1 / a^2 + 0 / b^2 = 1

/-- A parabola with focus at (1,0) and vertex at (m,0) -/
structure Parabola where
  m : ℝ

/-- The theorem statement -/
theorem ellipse_parabola_intersection (ε : Ellipse) (ρ : Parabola) :
  let e := Real.sqrt (1 - ε.b^2)
  (Real.sqrt (2/3) < e ∧ e < 1) →
  (1 < ρ.m ∧ ρ.m < (3 + Real.sqrt 2) / 4) := by
  sorry

end ellipse_parabola_intersection_l3873_387349


namespace trajectory_equation_l3873_387344

/-- The trajectory of point M(x, y) such that the ratio of its distance from the line x = 25/4
    to its distance from the point (4, 0) is 5/4 -/
theorem trajectory_equation (x y : ℝ) :
  (|x - 25/4| / Real.sqrt ((x - 4)^2 + y^2) = 5/4) →
  (x^2 / 25 + y^2 / 9 = 1) :=
by sorry

end trajectory_equation_l3873_387344


namespace max_notebooks_with_10_dollars_l3873_387399

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | single
  | pack4
  | pack7

/-- Returns the cost of a pack given its size -/
def packCost (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 3
  | PackSize.pack7 => 5

/-- Returns the number of notebooks in a pack given its size -/
def packNotebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 4
  | PackSize.pack7 => 7

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * packCost PackSize.single +
  p.pack4 * packCost PackSize.pack4 +
  p.pack7 * packCost PackSize.pack7

/-- Calculates the total number of notebooks in a purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * packNotebooks PackSize.single +
  p.pack4 * packNotebooks PackSize.pack4 +
  p.pack7 * packNotebooks PackSize.pack7

/-- The maximum number of notebooks that can be purchased with $10 is 14 -/
theorem max_notebooks_with_10_dollars :
  (∀ p : Purchase, totalCost p ≤ 10 → totalNotebooks p ≤ 14) ∧
  (∃ p : Purchase, totalCost p ≤ 10 ∧ totalNotebooks p = 14) :=
sorry

end max_notebooks_with_10_dollars_l3873_387399


namespace absolute_value_equation_solution_l3873_387390

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 10 :=
by
  -- The proof goes here
  sorry

end absolute_value_equation_solution_l3873_387390


namespace dodgeball_tournament_l3873_387338

theorem dodgeball_tournament (N : ℕ) : 
  (∃ W D : ℕ, 
    W + D = N * (N - 1) / 2 ∧ 
    15 * W + 22 * D = 1151) → 
  N = 12 := by
sorry

end dodgeball_tournament_l3873_387338


namespace marias_age_half_anns_l3873_387324

/-- Proves that Maria's age was half of Ann's age 4 years ago -/
theorem marias_age_half_anns (maria_current_age ann_current_age years_ago : ℕ) : 
  maria_current_age = 7 →
  ann_current_age = maria_current_age + 3 →
  maria_current_age - years_ago = (ann_current_age - years_ago) / 2 →
  years_ago = 4 := by
sorry

end marias_age_half_anns_l3873_387324


namespace beaver_home_fraction_l3873_387386

theorem beaver_home_fraction (total_beavers : ℕ) (swim_percentage : ℚ) :
  total_beavers = 4 →
  swim_percentage = 3/4 →
  (1 : ℚ) - swim_percentage = 1/4 :=
by sorry

end beaver_home_fraction_l3873_387386


namespace oliver_quarters_problem_l3873_387305

theorem oliver_quarters_problem (initial_cash : ℝ) (quarters_given : ℕ) (final_amount : ℝ) :
  initial_cash = 40 →
  quarters_given = 120 →
  final_amount = 55 →
  ∃ (Q : ℕ), 
    (initial_cash + 0.25 * Q) - (5 + 0.25 * quarters_given) = final_amount ∧
    Q = 200 :=
by sorry

end oliver_quarters_problem_l3873_387305


namespace g_invertible_interval_l3873_387308

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- State the theorem
theorem g_invertible_interval :
  ∃ (a : ℝ), a ≤ 2 ∧
  (∀ x y, a ≤ x ∧ x < y → g x < g y) ∧
  (∀ b, b < a → ∃ x y, b ≤ x ∧ x < y ∧ y < a ∧ g x ≥ g y) :=
sorry

end g_invertible_interval_l3873_387308


namespace solve_linear_equation_l3873_387389

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end solve_linear_equation_l3873_387389


namespace employee_gross_pay_l3873_387306

/-- Calculate the gross pay for an employee given regular and overtime hours and rates -/
def calculate_gross_pay (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (overtime_hours : ℚ) : ℚ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that the employee's gross pay for the week is $622 -/
theorem employee_gross_pay :
  let regular_rate : ℚ := 11.25
  let regular_hours : ℚ := 40
  let overtime_rate : ℚ := 16
  let overtime_hours : ℚ := 10.75
  calculate_gross_pay regular_rate regular_hours overtime_rate overtime_hours = 622 := by
  sorry

#eval calculate_gross_pay (11.25 : ℚ) (40 : ℚ) (16 : ℚ) (10.75 : ℚ)

end employee_gross_pay_l3873_387306


namespace geometric_sequence_sum_l3873_387330

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : a 1 + a 3 + a 5 = 21) 
    (h3 : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end geometric_sequence_sum_l3873_387330


namespace principal_is_2500_l3873_387369

/-- Given a simple interest, interest rate, and time period, calculates the principal amount. -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 2500. -/
theorem principal_is_2500 :
  let simple_interest : ℚ := 1000
  let rate : ℚ := 10
  let time : ℚ := 4
  calculate_principal simple_interest rate time = 2500 := by
  sorry

end principal_is_2500_l3873_387369


namespace parity_of_expression_l3873_387342

theorem parity_of_expression (o₁ o₂ n : ℤ) 
  (h₁ : ∃ k : ℤ, o₁ = 2*k + 1) 
  (h₂ : ∃ m : ℤ, o₂ = 2*m + 1) :
  (o₁^2 + n*o₁*o₂) % 2 = 1 ↔ n % 2 = 0 := by
  sorry

end parity_of_expression_l3873_387342
