import Mathlib

namespace jerry_action_figures_l1089_108972

theorem jerry_action_figures (total_needed : ℕ) (cost_per_figure : ℕ) (amount_needed : ℕ) :
  total_needed = 16 →
  cost_per_figure = 8 →
  amount_needed = 72 →
  total_needed - (amount_needed / cost_per_figure) = 7 :=
by sorry

end jerry_action_figures_l1089_108972


namespace gcd_1213_1985_l1089_108998

theorem gcd_1213_1985 : 
  (¬ (1213 % 2 = 0)) → 
  (¬ (1213 % 3 = 0)) → 
  (¬ (1213 % 5 = 0)) → 
  (¬ (1985 % 2 = 0)) → 
  (¬ (1985 % 3 = 0)) → 
  (¬ (1985 % 5 = 0)) → 
  Nat.gcd 1213 1985 = 1 := by
  sorry

end gcd_1213_1985_l1089_108998


namespace percentage_five_half_years_or_more_l1089_108997

/-- Represents the number of employees in each time period -/
structure EmployeeDistribution :=
  (less_than_half_year : ℕ)
  (half_to_one_year : ℕ)
  (one_to_one_half_years : ℕ)
  (one_half_to_two_years : ℕ)
  (two_to_two_half_years : ℕ)
  (two_half_to_three_years : ℕ)
  (three_to_three_half_years : ℕ)
  (three_half_to_four_years : ℕ)
  (four_to_four_half_years : ℕ)
  (four_half_to_five_years : ℕ)
  (five_to_five_half_years : ℕ)
  (five_half_to_six_years : ℕ)
  (six_to_six_half_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_half_year +
  d.half_to_one_year +
  d.one_to_one_half_years +
  d.one_half_to_two_years +
  d.two_to_two_half_years +
  d.two_half_to_three_years +
  d.three_to_three_half_years +
  d.three_half_to_four_years +
  d.four_to_four_half_years +
  d.four_half_to_five_years +
  d.five_to_five_half_years +
  d.five_half_to_six_years +
  d.six_to_six_half_years

/-- Calculates the number of employees working for 5.5 years or more -/
def employees_five_half_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.five_half_to_six_years + d.six_to_six_half_years

/-- Theorem stating that the percentage of employees working for 5.5 years or more is (2/38) * 100 -/
theorem percentage_five_half_years_or_more (d : EmployeeDistribution) 
  (h1 : d.less_than_half_year = 4)
  (h2 : d.half_to_one_year = 6)
  (h3 : d.one_to_one_half_years = 7)
  (h4 : d.one_half_to_two_years = 4)
  (h5 : d.two_to_two_half_years = 3)
  (h6 : d.two_half_to_three_years = 3)
  (h7 : d.three_to_three_half_years = 3)
  (h8 : d.three_half_to_four_years = 2)
  (h9 : d.four_to_four_half_years = 2)
  (h10 : d.four_half_to_five_years = 1)
  (h11 : d.five_to_five_half_years = 1)
  (h12 : d.five_half_to_six_years = 1)
  (h13 : d.six_to_six_half_years = 1) :
  (employees_five_half_years_or_more d : ℚ) / (total_employees d : ℚ) * 100 = 526 / 100 := by
  sorry

end percentage_five_half_years_or_more_l1089_108997


namespace delivery_speed_l1089_108932

/-- Given the conditions of the delivery problem, prove that the required average speed is 30 km/h -/
theorem delivery_speed (d : ℝ) (t : ℝ) (v : ℝ) : 
  (d / 60 = t - 1/4) →  -- Condition for moderate traffic
  (d / 20 = t + 1/4) →  -- Condition for traffic jams
  (d / v = 1/2) →       -- Condition for arriving exactly at 18:00
  v = 30 := by
  sorry

end delivery_speed_l1089_108932


namespace remaining_dogs_l1089_108964

theorem remaining_dogs (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 → dogs_given = 10 → 
  (10 : ℚ) / 27 * total_pets - dogs_given = 60 := by
sorry

end remaining_dogs_l1089_108964


namespace expansion_properties_l1089_108918

def n : ℕ := 5

def general_term (r : ℕ) : ℚ × ℤ → ℚ := λ (c, p) ↦ c * (2^(10 - r) * (-1)^r)

theorem expansion_properties :
  let tenth_term := general_term 9 (1, -8)
  let constant_term := general_term 5 (1, 0)
  let max_coeff_term := general_term 3 (1, 4)
  (tenth_term = -20) ∧
  (constant_term = -8064) ∧
  (max_coeff_term = -15360) ∧
  (∀ r : ℕ, r ≤ 10 → |general_term r (1, 10 - 2*r)| ≤ |max_coeff_term|) :=
by sorry

end expansion_properties_l1089_108918


namespace number_reading_and_approximation_l1089_108908

def number : ℕ := 60008205

def read_number (n : ℕ) : String := sorry

def approximate_to_ten_thousands (n : ℕ) : ℕ := sorry

theorem number_reading_and_approximation :
  (read_number number = "sixty million eight thousand two hundred and five") ∧
  (approximate_to_ten_thousands number = 6001) := by sorry

end number_reading_and_approximation_l1089_108908


namespace simplify_expression_l1089_108935

theorem simplify_expression : (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 := by
  sorry

end simplify_expression_l1089_108935


namespace point_coordinates_l1089_108943

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 5,
    and its distance to the y-axis is 3, then its coordinates are (3, -5) -/
theorem point_coordinates (p : Point) 
    (h1 : inFourthQuadrant p)
    (h2 : distanceToXAxis p = 5)
    (h3 : distanceToYAxis p = 3) :
    p = Point.mk 3 (-5) := by
  sorry

end point_coordinates_l1089_108943


namespace symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l1089_108979

/-- A function f: ℝ → ℝ is symmetric about the origin if f(-x) = -f(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem symmetric_implies_abs_even (f : ℝ → ℝ) :
  SymmetricAboutOrigin f → EvenFunction (fun x ↦ |f x|) :=
by sorry

theorem abs_even_not_sufficient_for_symmetric :
  ∃ f : ℝ → ℝ, EvenFunction (fun x ↦ |f x|) ∧ ¬SymmetricAboutOrigin f :=
by sorry

end symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l1089_108979


namespace merchandise_profit_analysis_l1089_108913

/-- Represents a store's merchandise sales model -/
structure MerchandiseModel where
  original_cost : ℝ
  original_price : ℝ
  original_sales : ℝ
  price_decrease_step : ℝ
  sales_increase_step : ℝ

/-- Calculate the profit given a price decrease -/
def profit (model : MerchandiseModel) (price_decrease : ℝ) : ℝ :=
  (model.original_sales + model.sales_increase_step * price_decrease) *
  (model.original_price - price_decrease - model.original_cost)

theorem merchandise_profit_analysis (model : MerchandiseModel)
  (h1 : model.original_cost = 80)
  (h2 : model.original_price = 100)
  (h3 : model.original_sales = 100)
  (h4 : model.price_decrease_step = 1)
  (h5 : model.sales_increase_step = 10) :
  profit model 0 = 2000 ∧
  (∀ x, profit model x = -10 * x^2 + 100 * x + 2000) ∧
  (∃ x, profit model x = 2250 ∧ ∀ y, profit model y ≤ profit model x) ∧
  (∀ p, 92 ≤ p ∧ p ≤ 98 ↔ profit model (100 - p) ≥ 2160) :=
by sorry

end merchandise_profit_analysis_l1089_108913


namespace intersection_in_fourth_quadrant_l1089_108971

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

-- Define a vertical line by a point
def VerticalLine (p : Point2D) :=
  {q : Point2D | q.x = p.x}

-- Define the intersection of two lines
def Intersection (l1 l2 : Set Point2D) :=
  {p : Point2D | p ∈ l1 ∧ p ∈ l2}

theorem intersection_in_fourth_quadrant :
  let l := Line ⟨-3, 0⟩ ⟨0, -5⟩
  let l' := VerticalLine ⟨2, 4⟩
  let i := Intersection l l'
  ∀ p ∈ i, p.x > 0 ∧ p.y < 0 := by sorry

end intersection_in_fourth_quadrant_l1089_108971


namespace arithmetic_sequence_n_values_l1089_108914

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1
  nth_term : ∀ n : ℕ, n ≥ 3 → a n = 70
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the possible values of n -/
theorem arithmetic_sequence_n_values (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≥ 3 ∧ seq.a n = 70 → n = 4 ∨ n = 24 ∨ n = 70 := by
  sorry

end arithmetic_sequence_n_values_l1089_108914


namespace intersection_unique_l1089_108968

/-- The system of linear equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  8 * x - 5 * y = 40 ∧ 6 * x + 2 * y = 14

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_unique :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point :=
by sorry

end intersection_unique_l1089_108968


namespace borrowing_interest_rate_l1089_108930

/-- Proves that the interest rate at which a person borrowed money is 4% per annum
    given the specified conditions. -/
theorem borrowing_interest_rate : 
  ∀ (principal : ℝ) (borrowing_time : ℝ) (lending_rate : ℝ) (lending_time : ℝ) (yearly_gain : ℝ),
  principal = 5000 →
  borrowing_time = 2 →
  lending_rate = 0.06 →
  lending_time = 2 →
  yearly_gain = 100 →
  ∃ (borrowing_rate : ℝ),
    borrowing_rate = 0.04 ∧
    principal * lending_rate * lending_time - 
    principal * borrowing_rate * borrowing_time = 
    yearly_gain * borrowing_time :=
by sorry

end borrowing_interest_rate_l1089_108930


namespace billion_scientific_notation_l1089_108903

theorem billion_scientific_notation : 
  (1.1 * 10^9 : ℝ) = 1100000000 := by sorry

end billion_scientific_notation_l1089_108903


namespace johns_break_time_l1089_108939

/-- Given the dancing times of John and James, prove that John's break was 1 hour long. -/
theorem johns_break_time (john_first_dance : ℝ) (john_second_dance : ℝ) 
  (james_dance_multiplier : ℝ) (total_dance_time : ℝ) :
  john_first_dance = 3 →
  john_second_dance = 5 →
  james_dance_multiplier = 1/3 →
  total_dance_time = 20 →
  ∃ (break_time : ℝ),
    total_dance_time = john_first_dance + john_second_dance + 
      ((john_first_dance + john_second_dance + break_time) + 
       james_dance_multiplier * (john_first_dance + john_second_dance + break_time)) ∧
    break_time = 1 := by
  sorry


end johns_break_time_l1089_108939


namespace complex_expression_simplification_l1089_108948

theorem complex_expression_simplification (x y : ℝ) :
  let i : ℂ := Complex.I
  (x^2 + i*y)^3 * (x^2 - i*y)^3 = x^12 - 9*x^8*y^2 - 9*x^4*y^4 - y^6 :=
by sorry

end complex_expression_simplification_l1089_108948


namespace unique_real_sqrt_negative_square_l1089_108926

theorem unique_real_sqrt_negative_square :
  ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(2 * x - 3) ^ 2 := by
  sorry

end unique_real_sqrt_negative_square_l1089_108926


namespace inequality_proof_l1089_108941

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end inequality_proof_l1089_108941


namespace min_people_to_remove_l1089_108977

def total_people : ℕ := 73

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def people_to_remove (n : ℕ) : ℕ := total_people - n

theorem min_people_to_remove :
  ∃ n : ℕ, is_square n ∧
    (∀ m : ℕ, is_square m → people_to_remove m ≥ people_to_remove n) ∧
    people_to_remove n = 9 :=
by sorry

end min_people_to_remove_l1089_108977


namespace factorization_of_x_squared_minus_one_l1089_108924

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_of_x_squared_minus_one_l1089_108924


namespace triangle_sum_maximum_l1089_108991

theorem triangle_sum_maximum (arrangement : List ℕ) : 
  (arrangement.toFinset = Finset.range 9 \ {0}) →
  (∃ (side1 side2 side3 : List ℕ), 
    side1.length = 4 ∧ side2.length = 4 ∧ side3.length = 4 ∧
    (side1 ++ side2 ++ side3).toFinset = arrangement.toFinset ∧
    side1.sum = side2.sum ∧ side2.sum = side3.sum) →
  (∀ (side : List ℕ), side.toFinset ⊆ arrangement.toFinset ∧ side.length = 4 → side.sum ≤ 19) :=
by sorry

#check triangle_sum_maximum

end triangle_sum_maximum_l1089_108991


namespace evaluate_expression_l1089_108975

theorem evaluate_expression (a b : ℕ+) (h : 2^(a:ℕ) * 3^(b:ℕ) = 324) : 
  2^(b:ℕ) * 3^(a:ℕ) = 144 := by
  sorry

end evaluate_expression_l1089_108975


namespace stratified_sample_theorem_l1089_108982

/-- Represents the number of athletes selected in a stratified sample -/
structure StratifiedSample where
  totalMale : ℕ
  totalFemale : ℕ
  selectedMale : ℕ
  selectedFemale : ℕ

/-- Checks if the sample maintains the same ratio as the total population -/
def isProportionalSample (s : StratifiedSample) : Prop :=
  s.totalMale * s.selectedFemale = s.totalFemale * s.selectedMale

/-- Theorem: Given the conditions, the number of selected female athletes is 6 -/
theorem stratified_sample_theorem (s : StratifiedSample) :
  s.totalMale = 56 →
  s.totalFemale = 42 →
  s.selectedMale = 8 →
  isProportionalSample s →
  s.selectedFemale = 6 := by
  sorry

#check stratified_sample_theorem

end stratified_sample_theorem_l1089_108982


namespace crepe_myrtle_count_l1089_108921

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white = total - (pink + red) →
  white = 26 := by
sorry

end crepe_myrtle_count_l1089_108921


namespace unfactorable_quadratic_l1089_108940

theorem unfactorable_quadratic : ¬ ∃ (a b c : ℝ), (∀ x : ℝ, x^2 - 10*x - 25 = (a*x + b)*(c*x + 1)) := by
  sorry

end unfactorable_quadratic_l1089_108940


namespace amy_biking_distance_l1089_108958

theorem amy_biking_distance (x : ℝ) : 
  x + (2 * x - 3) = 33 → x = 12 := by sorry

end amy_biking_distance_l1089_108958


namespace cubic_equation_root_l1089_108915

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0 → 
  d = -37/2 := by
sorry

end cubic_equation_root_l1089_108915


namespace arith_seq_mono_increasing_iff_a2_gt_a1_l1089_108953

/-- An arithmetic sequence -/
def ArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonoIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- Theorem: For an arithmetic sequence, a_2 > a_1 is equivalent to the sequence being monotonically increasing -/
theorem arith_seq_mono_increasing_iff_a2_gt_a1 (a : ℕ → ℝ) (h : ArithmeticSeq a) :
  a 2 > a 1 ↔ MonoIncreasing a := by sorry

end arith_seq_mono_increasing_iff_a2_gt_a1_l1089_108953


namespace wendy_album_problem_l1089_108990

/-- Given a total number of pictures and the number of pictures in each of 5 albums,
    calculate the number of pictures in the first album. -/
def pictures_in_first_album (total : ℕ) (pics_per_album : ℕ) : ℕ :=
  total - 5 * pics_per_album

/-- Theorem stating that given 79 total pictures and 7 pictures in each of 5 albums,
    the number of pictures in the first album is 44. -/
theorem wendy_album_problem :
  pictures_in_first_album 79 7 = 44 := by
  sorry

end wendy_album_problem_l1089_108990


namespace shoe_refund_percentage_l1089_108984

/-- Given Will's shopping scenario, prove the percentage of shoe price refunded --/
theorem shoe_refund_percentage 
  (initial_amount : ℝ) 
  (sweater_cost : ℝ) 
  (tshirt_cost : ℝ) 
  (shoe_cost : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_amount = 74) 
  (h2 : sweater_cost = 9) 
  (h3 : tshirt_cost = 11) 
  (h4 : shoe_cost = 30) 
  (h5 : final_amount = 51) : 
  (final_amount - (initial_amount - (sweater_cost + tshirt_cost + shoe_cost))) / shoe_cost * 100 = 90 := by
  sorry

end shoe_refund_percentage_l1089_108984


namespace smallest_group_size_exists_group_size_l1089_108944

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) → n ≥ 187 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) ∧ n = 187 :=
by sorry

end smallest_group_size_exists_group_size_l1089_108944


namespace parallel_heaters_boiling_time_l1089_108910

/-- Given two heaters connected to the same direct current source,
    prove that the time to boil water when connected in parallel
    is (t₁ * t₂) / (t₁ + t₂), where t₁ and t₂ are the times taken
    by each heater individually. -/
theorem parallel_heaters_boiling_time
  (t₁ t₂ : ℝ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (boil_time : ℝ → ℝ → ℝ) :
  boil_time t₁ t₂ = t₁ * t₂ / (t₁ + t₂) :=
by sorry

end parallel_heaters_boiling_time_l1089_108910


namespace sqrt_inequality_and_trig_identity_l1089_108949

theorem sqrt_inequality_and_trig_identity :
  (∀ (α : Real),
    Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 ∧
    Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4) :=
by sorry

end sqrt_inequality_and_trig_identity_l1089_108949


namespace combined_gold_cost_l1089_108911

/-- The cost of Gary and Anna's combined gold -/
theorem combined_gold_cost (gary_grams anna_grams : ℕ) (gary_price anna_price : ℚ) : 
  gary_grams = 30 → 
  gary_price = 15 → 
  anna_grams = 50 → 
  anna_price = 20 → 
  gary_grams * gary_price + anna_grams * anna_price = 1450 := by
  sorry


end combined_gold_cost_l1089_108911


namespace class_payment_problem_l1089_108907

theorem class_payment_problem (total_students : ℕ) (full_payers : ℕ) (half_payers : ℕ) (total_amount : ℕ) :
  total_students = 25 →
  full_payers = 21 →
  half_payers = 4 →
  total_amount = 1150 →
  ∃ (full_payment : ℕ), 
    full_payment * full_payers + (full_payment / 2) * half_payers = total_amount ∧
    full_payment = 50 :=
by
  sorry

end class_payment_problem_l1089_108907


namespace quadratic_no_real_roots_l1089_108970

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 - 3*x + 3 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l1089_108970


namespace different_subjects_count_l1089_108901

/-- The number of ways to choose 2 books from different subjects -/
def choose_different_subjects (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that there are 143 ways to choose 2 books from different subjects -/
theorem different_subjects_count :
  choose_different_subjects 9 7 5 = 143 := by
  sorry

end different_subjects_count_l1089_108901


namespace absolute_value_of_negative_l1089_108947

theorem absolute_value_of_negative (a : ℝ) : a < 0 → |a| = -a := by
  sorry

end absolute_value_of_negative_l1089_108947


namespace quadratic_monotone_increasing_l1089_108951

/-- A quadratic function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem quadratic_monotone_increasing (a : ℝ) (h : a > 1) :
  ∀ x > 1, Monotone (fun x => f a x) := by sorry

end quadratic_monotone_increasing_l1089_108951


namespace select_four_from_fifteen_l1089_108996

theorem select_four_from_fifteen (n : ℕ) (h : n = 15) :
  (n * (n - 1) * (n - 2) * (n - 3)) = 32760 := by
  sorry

end select_four_from_fifteen_l1089_108996


namespace unreachable_corner_l1089_108963

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- The set of 7 vertices of a cube, excluding (1,1,1) -/
def cube_vertices : Set Point3D :=
  { ⟨0,0,0⟩, ⟨0,0,1⟩, ⟨0,1,0⟩, ⟨1,0,0⟩, ⟨0,1,1⟩, ⟨1,0,1⟩, ⟨1,1,0⟩ }

/-- Symmetry transformation with respect to another point -/
def symmetry_transform (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

/-- The set of points reachable through symmetry transformations -/
def reachable_points : Set Point3D :=
  sorry -- Definition of reachable points through symmetry transformations

theorem unreachable_corner : ⟨1,1,1⟩ ∉ reachable_points := by
  sorry

#check unreachable_corner

end unreachable_corner_l1089_108963


namespace eccentricity_is_sqrt_three_l1089_108999

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A circle centered on a hyperbola and tangent to the x-axis at a focus -/
structure TangentCircle (h : Hyperbola) where
  center : ℝ × ℝ
  h_on_hyperbola : center.1^2 / h.a^2 - center.2^2 / h.b^2 = 1
  h_tangent_at_focus : center.1 = h.a * (h.a^2 + h.b^2).sqrt / h.a

/-- The property that the circle intersects the y-axis forming an equilateral triangle -/
def forms_equilateral_triangle (h : Hyperbola) (c : TangentCircle h) : Prop :=
  ∃ (y₁ y₂ : ℝ), 
    y₁ < c.center.2 ∧ c.center.2 < y₂ ∧
    (c.center.1^2 + (y₁ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (c.center.1^2 + (y₂ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (y₂ - y₁ = h.b^2 / h.a * Real.sqrt 3)

/-- The theorem stating that the eccentricity of the hyperbola is √3 -/
theorem eccentricity_is_sqrt_three (h : Hyperbola) (c : TangentCircle h)
  (h_equilateral : forms_equilateral_triangle h c) :
  (h.a^2 + h.b^2).sqrt / h.a = Real.sqrt 3 := by
  sorry

end eccentricity_is_sqrt_three_l1089_108999


namespace condition_necessary_not_sufficient_l1089_108919

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for a geometric sequence. -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The condition given in the problem. -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) :=
sorry

end condition_necessary_not_sufficient_l1089_108919


namespace expression_evaluation_l1089_108974

theorem expression_evaluation (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end expression_evaluation_l1089_108974


namespace harper_gift_cost_l1089_108955

/-- The total amount spent on teacher appreciation gifts --/
def total_gift_cost (son_teachers daughter_teachers gift_cost : ℕ) : ℕ :=
  (son_teachers + daughter_teachers) * gift_cost

/-- Theorem: Harper's total gift cost is $70 --/
theorem harper_gift_cost :
  total_gift_cost 3 4 10 = 70 := by
  sorry

end harper_gift_cost_l1089_108955


namespace high_card_value_l1089_108965

structure CardGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_value : Nat
  low_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_earn : Nat

def is_valid_game (game : CardGame) : Prop :=
  game.total_cards = 52 ∧
  game.high_cards = game.low_cards ∧
  game.high_cards + game.low_cards = game.total_cards ∧
  game.low_value = 1 ∧
  game.target_points = 5 ∧
  game.target_low_cards = 3 ∧
  game.ways_to_earn = 4

theorem high_card_value (game : CardGame) :
  is_valid_game game → game.high_value = 2 := by
  sorry

end high_card_value_l1089_108965


namespace parallel_vectors_l1089_108978

def a : ℝ × ℝ := (1, -1)
def b : ℝ → ℝ × ℝ := λ x => (x, 1)

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, b x = k • a) → x = -1 := by sorry

end parallel_vectors_l1089_108978


namespace slant_height_angle_is_30_degrees_l1089_108934

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Side length of the base square -/
  base_side : ℝ
  /-- Angle between lateral face and base plane -/
  lateral_angle : ℝ

/-- Angle between slant height and adjacent face -/
def slant_height_angle (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem slant_height_angle_is_30_degrees (p : RegularQuadPyramid) 
  (h : p.lateral_angle = Real.pi / 4) : 
  slant_height_angle p = Real.pi / 6 := by
  sorry

end slant_height_angle_is_30_degrees_l1089_108934


namespace number_of_teams_l1089_108976

/-- The number of teams in the league -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 4900

/-- Each team faces every other team this many times -/
def games_per_pair : ℕ := 4

theorem number_of_teams : 
  (n * games_per_pair * (n - 1)) / 2 = total_games ∧ n = 50 := by sorry

end number_of_teams_l1089_108976


namespace intersection_of_P_and_Q_l1089_108983

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≥ 1} := by
  sorry

end intersection_of_P_and_Q_l1089_108983


namespace sin_90_degrees_l1089_108950

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l1089_108950


namespace percentage_4_plus_years_l1089_108912

/-- Represents the number of employees in each year group -/
structure EmployeeDistribution :=
  (less_than_1 : ℕ)
  (one_to_2 : ℕ)
  (two_to_3 : ℕ)
  (three_to_4 : ℕ)
  (four_to_5 : ℕ)
  (five_to_6 : ℕ)
  (six_to_7 : ℕ)
  (seven_to_8 : ℕ)
  (eight_to_9 : ℕ)
  (nine_to_10 : ℕ)
  (ten_plus : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1 + d.one_to_2 + d.two_to_3 + d.three_to_4 + d.four_to_5 + 
  d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Calculates the number of employees who have worked for 4 years or more -/
def employees_4_plus_years (d : EmployeeDistribution) : ℕ :=
  d.four_to_5 + d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Theorem: The percentage of employees who have worked for 4 years or more is 37.5% -/
theorem percentage_4_plus_years (d : EmployeeDistribution) : 
  (employees_4_plus_years d : ℚ) / (total_employees d : ℚ) = 375 / 1000 :=
sorry


end percentage_4_plus_years_l1089_108912


namespace unique_prime_square_solution_l1089_108989

theorem unique_prime_square_solution :
  ∀ (p m : ℕ), 
    Prime p → 
    m > 0 → 
    2 * p^2 + p + 9 = m^2 → 
    p = 5 ∧ m = 8 := by
  sorry

end unique_prime_square_solution_l1089_108989


namespace solution_set_eq_singleton_l1089_108962

/-- The set of solutions to the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set is equal to {(1, 1)} -/
theorem solution_set_eq_singleton : solution_set = {(1, 1)} := by
  sorry

#check solution_set_eq_singleton

end solution_set_eq_singleton_l1089_108962


namespace equation_solution_l1089_108933

theorem equation_solution :
  ∀ x : ℚ, x ≠ 2 →
  (7 * x / (x - 2) - 5 / (x - 2) = 3 / (x - 2)) ↔ x = 8 / 7 := by
sorry

end equation_solution_l1089_108933


namespace quadratic_integer_solution_count_l1089_108994

theorem quadratic_integer_solution_count : ∃ (S : Finset ℚ),
  (∀ k ∈ S, |k| < 100 ∧ ∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) ∧
  (∀ k : ℚ, |k| < 100 → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) → k ∈ S) ∧
  Finset.card S = 40 :=
sorry

end quadratic_integer_solution_count_l1089_108994


namespace complex_repair_charge_correct_l1089_108980

/-- Calculates the charge for a complex bike repair given the following conditions:
  * Tire repair charge is $20
  * Tire repair cost is $5
  * Number of tire repairs in a month is 300
  * Number of complex repairs in a month is 2
  * Complex repair cost is $50
  * Retail profit is $2000
  * Fixed expenses are $4000
  * Total monthly profit is $3000
-/
def complex_repair_charge (
  tire_repair_charge : ℕ)
  (tire_repair_cost : ℕ)
  (num_tire_repairs : ℕ)
  (num_complex_repairs : ℕ)
  (complex_repair_cost : ℕ)
  (retail_profit : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ) : ℕ :=
  let tire_repair_profit := (tire_repair_charge - tire_repair_cost) * num_tire_repairs
  let total_profit_before_complex := tire_repair_profit + retail_profit - fixed_expenses
  let complex_repair_total_profit := total_profit - total_profit_before_complex
  let complex_repair_profit := complex_repair_total_profit / num_complex_repairs
  complex_repair_profit + complex_repair_cost

theorem complex_repair_charge_correct : 
  complex_repair_charge 20 5 300 2 50 2000 4000 3000 = 300 := by
  sorry

end complex_repair_charge_correct_l1089_108980


namespace system_solution_l1089_108986

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 2 ∧ x - 2*y = 6) ↔ (x = 4 ∧ y = -1) :=
by sorry

end system_solution_l1089_108986


namespace quadratic_sets_problem_l1089_108966

theorem quadratic_sets_problem (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + 15 = 0}
  let B := {x : ℝ | x^2 - 5*x + q = 0}
  (A ∩ B = {3}) →
  (p = -8 ∧ q = 6 ∧ A ∪ B = {2, 3, 5}) :=
by sorry

end quadratic_sets_problem_l1089_108966


namespace quadratic_function_properties_l1089_108929

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) (h1 : 1/3 ≤ a) (h2 : a ≤ 1) :
  -- 1. Minimum value of f(x) on [1, 3]
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 3 → f a x ≤ f a y) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ f a x = 1 - 1/a) ∧
  -- 2. Minimum value of M(a) - N(a)
  (∃ (M N : ℝ → ℝ),
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → f a x ≤ M a) ∧
    (∃ (y : ℝ), y ∈ Set.Icc 1 3 ∧ f a y = M a) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → N a ≤ f a x) ∧
    (∃ (z : ℝ), z ∈ Set.Icc 1 3 ∧ f a z = N a) ∧
    (∀ (b : ℝ), 1/3 ≤ b ∧ b ≤ 1 → 1/2 ≤ M b - N b) ∧
    (∃ (c : ℝ), 1/3 ≤ c ∧ c ≤ 1 ∧ M c - N c = 1/2)) := by
  sorry

end quadratic_function_properties_l1089_108929


namespace digit_sum_problem_l1089_108985

theorem digit_sum_problem :
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    22 * (a + b + c) = 462 →
    ((a = 4 ∧ b = 8 ∧ c = 9) ∨
     (a = 5 ∧ b = 7 ∧ c = 9) ∨
     (a = 6 ∧ b = 7 ∧ c = 8) ∨
     (a = 8 ∧ b = 4 ∧ c = 9) ∨
     (a = 7 ∧ b = 5 ∧ c = 9) ∨
     (a = 7 ∧ b = 6 ∧ c = 8) ∨
     (a = 9 ∧ b = 4 ∧ c = 8) ∨
     (a = 9 ∧ b = 5 ∧ c = 7) ∨
     (a = 8 ∧ b = 6 ∧ c = 7)) :=
by sorry

end digit_sum_problem_l1089_108985


namespace lemonade_stand_total_profit_l1089_108936

/-- Calculates the profit for a single day of lemonade stand operation -/
def daily_profit (lemon_cost sugar_cost cup_cost extra_cost price_per_cup cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost + extra_cost)

/-- Represents the lemonade stand operation for three days -/
def lemonade_stand_profit : Prop :=
  let day1_profit := daily_profit 10 5 3 0 4 21
  let day2_profit := daily_profit 12 6 4 0 5 18
  let day3_profit := daily_profit 8 4 3 2 4 25
  day1_profit + day2_profit + day3_profit = 217

theorem lemonade_stand_total_profit : lemonade_stand_profit := by
  sorry

end lemonade_stand_total_profit_l1089_108936


namespace fibonacci_special_sequence_l1089_108954

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_special_sequence (a b c : ℕ) :
  (fib c = 2 * fib b - fib a) →
  (fib c - fib a = fib a) →
  (a + c = 1700) →
  a = 849 := by
  sorry

end fibonacci_special_sequence_l1089_108954


namespace quadratic_root_one_iff_sum_zero_l1089_108906

theorem quadratic_root_one_iff_sum_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end quadratic_root_one_iff_sum_zero_l1089_108906


namespace binomial_expansion_properties_l1089_108967

theorem binomial_expansion_properties :
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  -- The coefficient of the third term
  (Finset.sum (Finset.range 1) (fun k => (n.choose k) * a^(n-k) * b^k)) = 40 ∧
  -- The sum of all binomial coefficients
  (Finset.sum (Finset.range (n+1)) (fun k => n.choose k)) = 32 := by
sorry

end binomial_expansion_properties_l1089_108967


namespace least_positive_integer_divisible_by_four_primes_l1089_108904

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ * p₂ * p₃ * p₄ = n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ * q₂ * q₃ * q₄ = m)) ∧
  n = 210 := by
sorry

end least_positive_integer_divisible_by_four_primes_l1089_108904


namespace red_white_jelly_beans_in_fishbowl_l1089_108902

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_bowl : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem red_white_jelly_beans_in_fishbowl :
  total_red_white_in_bowl = 126 :=
by sorry

end red_white_jelly_beans_in_fishbowl_l1089_108902


namespace intersection_equals_open_unit_interval_l1089_108922

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_open_unit_interval : M ∩ N = open_unit_interval := by
  sorry

end intersection_equals_open_unit_interval_l1089_108922


namespace cube_volume_from_space_diagonal_l1089_108992

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by
  sorry

end cube_volume_from_space_diagonal_l1089_108992


namespace claire_balloons_l1089_108931

/-- The number of balloons Claire has at the end of the fair --/
def final_balloon_count (initial : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away : ℕ) (taken_from_coworker : ℕ) : ℕ :=
  initial - given_to_girl - floated_away - given_away + taken_from_coworker

/-- Theorem stating that Claire ends up with 39 balloons --/
theorem claire_balloons : 
  final_balloon_count 50 1 12 9 11 = 39 := by
  sorry

end claire_balloons_l1089_108931


namespace range_of_f_l1089_108942

def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 := by sorry

end range_of_f_l1089_108942


namespace gcd_45_105_l1089_108920

theorem gcd_45_105 : Nat.gcd 45 105 = 15 := by
  sorry

end gcd_45_105_l1089_108920


namespace cube_sum_theorem_l1089_108946

theorem cube_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : a * b + a * c + b * c = 6)
  (prod_eq : a * b * c = -8) :
  a^3 + b^3 + c^3 = 8 := by
sorry

end cube_sum_theorem_l1089_108946


namespace a_less_than_two_necessary_and_sufficient_l1089_108905

theorem a_less_than_two_necessary_and_sufficient (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x| > a) ↔ a < 2 := by
  sorry

end a_less_than_two_necessary_and_sufficient_l1089_108905


namespace point_transformation_l1089_108927

def rotate_270_clockwise (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate_270_clockwise a b 2 3
  let (x₂, y₂) := reflect_about_y_eq_x x₁ y₁
  (x₂ = 4 ∧ y₂ = -7) → b - a = -7 := by
  sorry

end point_transformation_l1089_108927


namespace x_over_y_value_l1089_108969

theorem x_over_y_value (x y : ℝ) (h1 : x * y = 1) (h2 : x > 0) (h3 : y > 0) (h4 : y = 0.16666666666666666) :
  x / y = 36 := by
  sorry

end x_over_y_value_l1089_108969


namespace function_minimum_condition_l1089_108987

/-- A function f(x) = x^2 - 2ax + a has a minimum value in the interval (-∞, 1) if and only if a < 1 -/
theorem function_minimum_condition (a : ℝ) : 
  (∃ (x₀ : ℝ), x₀ < 1 ∧ ∀ (x : ℝ), x < 1 → (x^2 - 2*a*x + a) ≥ (x₀^2 - 2*a*x₀ + a)) ↔ a < 1 := by
  sorry

end function_minimum_condition_l1089_108987


namespace inequality_proof_l1089_108925

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l1089_108925


namespace preimage_of_5_1_l1089_108995

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that the pre-image of (5, 1) under f is (2, 3) -/
theorem preimage_of_5_1 : f (2, 3) = (5, 1) := by sorry

end preimage_of_5_1_l1089_108995


namespace power_division_l1089_108988

theorem power_division (x : ℝ) (h : x ≠ 0) : x^8 / x^2 = x^6 := by
  sorry

end power_division_l1089_108988


namespace det_A_nonzero_l1089_108923

def matrix_A (n : ℕ) (a : ℤ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => a^(i.val * j.val + 1)

theorem det_A_nonzero {n : ℕ} {a : ℤ} (h : a > 1) :
  Matrix.det (matrix_A n a) ≠ 0 := by
  sorry

end det_A_nonzero_l1089_108923


namespace parallel_lines_symmetry_intersecting_lines_symmetry_l1089_108909

-- Define a type for lines in a plane
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a type for points in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if two lines intersect
def intersect (l1 l2 : Line2D) : Prop :=
  ¬(are_parallel l1 l2)

-- Define a type for axis of symmetry
structure AxisOfSymmetry where
  line : Line2D

-- Define a type for center of symmetry
structure CenterOfSymmetry where
  point : Point2D

-- Theorem for parallel lines
theorem parallel_lines_symmetry (l1 l2 : Line2D) (h : are_parallel l1 l2) :
  ∃ (axis : AxisOfSymmetry), (∀ (center : CenterOfSymmetry), True) :=
sorry

-- Theorem for intersecting lines
theorem intersecting_lines_symmetry (l1 l2 : Line2D) (h : intersect l1 l2) :
  ∃ (axis1 axis2 : AxisOfSymmetry) (center : CenterOfSymmetry),
    axis1.line.slope * axis2.line.slope = -1 :=
sorry

end parallel_lines_symmetry_intersecting_lines_symmetry_l1089_108909


namespace ship_speed_in_still_water_l1089_108938

/-- Given a ship with downstream speed of 32 km/h and upstream speed of 28 km/h,
    prove that its speed in still water is 30 km/h. -/
theorem ship_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 32)
  (h2 : upstream_speed = 28)
  (h3 : ∃ (ship_speed stream_speed : ℝ), 
    ship_speed > stream_speed ∧
    ship_speed + stream_speed = downstream_speed ∧
    ship_speed - stream_speed = upstream_speed) :
  ∃ (ship_speed : ℝ), ship_speed = 30 := by
sorry

end ship_speed_in_still_water_l1089_108938


namespace arithmetic_calculations_l1089_108900

theorem arithmetic_calculations :
  ((-53 + 21 + 79 - 37) = 10) ∧
  ((-9 - 1/3 - (abs (-4 - 5/6)) + (abs (0 - 5 - 1/6)) - 2/3) = -29/3) ∧
  ((-2^3 * (-4)^2 / (4/3) + abs (5 - 8)) = -93) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) := by
  sorry

end arithmetic_calculations_l1089_108900


namespace arithmetic_sequence_common_difference_l1089_108981

/-- 
Given an arithmetic sequence {aₙ} with common difference d,
prove that d = 1 when S₈ = 8a₅ - 4, where Sₙ is the sum of the first n terms.
-/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (d : ℚ)      -- The common difference
  (S : ℕ → ℚ)  -- The sum function
  (h1 : ∀ n, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  (h2 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)  -- Sum formula for arithmetic sequence
  (h3 : S 8 = 8 * a 5 - 4)  -- Given condition
  : d = 1 := by
sorry

end arithmetic_sequence_common_difference_l1089_108981


namespace rectangleABCD_area_is_196_l1089_108961

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of rectangle ABCD formed by three identical rectangles -/
def rectangleABCD_area (small_rect : Rectangle) : ℝ :=
  (2 * small_rect.width) * small_rect.length

theorem rectangleABCD_area_is_196 (small_rect : Rectangle) 
  (h1 : small_rect.width = 7)
  (h2 : small_rect.length = 2 * small_rect.width) :
  rectangleABCD_area small_rect = 196 := by
  sorry

#eval rectangleABCD_area { width := 7, length := 14 }

end rectangleABCD_area_is_196_l1089_108961


namespace percentage_puppies_greater_profit_l1089_108959

/-- Calculates the percentage of puppies that can be sold for a greater profit -/
theorem percentage_puppies_greater_profit (total_puppies : ℕ) (puppies_more_than_4_spots : ℕ) 
  (h1 : total_puppies = 10)
  (h2 : puppies_more_than_4_spots = 6) :
  (puppies_more_than_4_spots : ℚ) / total_puppies * 100 = 60 := by
  sorry


end percentage_puppies_greater_profit_l1089_108959


namespace odd_cube_difference_divisible_by_24_l1089_108957

theorem odd_cube_difference_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1)^3 - (2 * n + 1)) := by sorry

end odd_cube_difference_divisible_by_24_l1089_108957


namespace difference_of_squares_l1089_108952

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end difference_of_squares_l1089_108952


namespace m_value_min_value_l1089_108916

-- Define the solution set A
def A (m : ℤ) : Set ℝ := {x : ℝ | |x + 1| + |x - m| < 5}

-- Theorem 1
theorem m_value (m : ℤ) (h : 3 ∈ A m) : m = 3 := by sorry

-- Theorem 2
theorem min_value (a b c : ℝ) (h : a + 2*b + 2*c = 3) : 
  ∃ (min : ℝ), min = 1 ∧ a^2 + b^2 + c^2 ≥ min := by sorry

end m_value_min_value_l1089_108916


namespace fourth_power_equality_l1089_108917

theorem fourth_power_equality (x : ℝ) : x^4 = (-3)^4 → x = 3 ∨ x = -3 := by
  sorry

end fourth_power_equality_l1089_108917


namespace f_increasing_iff_a_in_range_l1089_108945

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for a function to be increasing
def IsIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  IsIncreasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end f_increasing_iff_a_in_range_l1089_108945


namespace distributor_cost_l1089_108973

theorem distributor_cost (commission_rate : Real) (profit_rate : Real) (observed_price : Real) :
  commission_rate = 0.20 →
  profit_rate = 0.20 →
  observed_price = 30 →
  ∃ (cost : Real),
    cost = 31.25 ∧
    observed_price = (1 - commission_rate) * (cost * (1 + profit_rate)) :=
by sorry

end distributor_cost_l1089_108973


namespace y_intercept_for_specific_line_l1089_108960

/-- A line in 2D space with a given slope and x-intercept. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ := l.slope * (-l.x_intercept) + 0

/-- Theorem stating that a line with slope -3 and x-intercept (3, 0) has y-intercept (0, 9). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 3 }
  y_intercept l = 9 := by
  sorry


end y_intercept_for_specific_line_l1089_108960


namespace num_paths_5x4_grid_l1089_108937

/-- The number of paths on a grid from point C to point D -/
def num_paths (grid_width grid_height path_length right_steps up_steps : ℕ) : ℕ :=
  Nat.choose path_length up_steps

/-- Theorem stating the number of paths on a 5x4 grid with specific constraints -/
theorem num_paths_5x4_grid : num_paths 5 4 8 5 3 = 56 := by
  sorry

end num_paths_5x4_grid_l1089_108937


namespace complex_equation_product_l1089_108956

theorem complex_equation_product (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a * b = -2 := by
  sorry

end complex_equation_product_l1089_108956


namespace fred_has_61_cards_l1089_108928

/-- The number of baseball cards Fred has after all transactions -/
def fred_final_cards (initial : ℕ) (given_to_mary : ℕ) (found_in_box : ℕ) (given_to_john : ℕ) (purchased : ℕ) : ℕ :=
  initial - given_to_mary + found_in_box - given_to_john + purchased

/-- Theorem stating that Fred ends up with 61 cards -/
theorem fred_has_61_cards :
  fred_final_cards 26 18 40 12 25 = 61 := by
  sorry

end fred_has_61_cards_l1089_108928


namespace daily_production_l1089_108993

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 5

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 12000

/-- The total number of bottles produced per day -/
def total_bottles : ℕ := bottles_per_case * cases_per_day

theorem daily_production :
  total_bottles = 60000 :=
by sorry

end daily_production_l1089_108993
