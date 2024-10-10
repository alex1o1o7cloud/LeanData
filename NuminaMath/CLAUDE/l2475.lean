import Mathlib

namespace f_min_value_l2475_247555

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

-- State the theorem
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) :=
sorry

end f_min_value_l2475_247555


namespace can_obtain_11_from_1_l2475_247550

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | MultiplyBy3
  | Add3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.DivideBy3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem stating that 11 can be obtained from 1 using the allowed operations -/
theorem can_obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 :=
  sorry

end can_obtain_11_from_1_l2475_247550


namespace equal_share_amount_l2475_247581

def emani_money : ℕ := 150
def howard_money : ℕ := emani_money - 30

def total_money : ℕ := emani_money + howard_money
def shared_amount : ℕ := total_money / 2

theorem equal_share_amount :
  shared_amount = 135 := by sorry

end equal_share_amount_l2475_247581


namespace luncheon_tables_l2475_247569

def tables_needed (invited : ℕ) (no_show : ℕ) (seats_per_table : ℕ) : ℕ :=
  ((invited - no_show) + seats_per_table - 1) / seats_per_table

theorem luncheon_tables :
  tables_needed 47 7 5 = 8 :=
by
  sorry

end luncheon_tables_l2475_247569


namespace number_of_factors_of_M_l2475_247579

def M : ℕ := 57^5 + 5*57^4 + 10*57^3 + 10*57^2 + 5*57 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 36 :=
by sorry

end number_of_factors_of_M_l2475_247579


namespace triangle_inequality_l2475_247574

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
sorry

end triangle_inequality_l2475_247574


namespace visible_friends_count_l2475_247575

theorem visible_friends_count : 
  (Finset.sum (Finset.range 10) (λ i => 
    (Finset.filter (λ j => Nat.gcd (i + 1) j = 1) (Finset.range 6)).card
  )) + 10 = 36 := by
  sorry

end visible_friends_count_l2475_247575


namespace red_papers_count_l2475_247509

theorem red_papers_count (papers_per_box : ℕ) (num_boxes : ℕ) : 
  papers_per_box = 2 → num_boxes = 2 → papers_per_box * num_boxes = 4 := by
  sorry

end red_papers_count_l2475_247509


namespace sin_greater_cos_range_l2475_247549

theorem sin_greater_cos_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) := by
  sorry

end sin_greater_cos_range_l2475_247549


namespace maria_towel_problem_l2475_247593

/-- Represents the number of towels Maria has -/
structure TowelCount where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the total number of towels -/
def TowelCount.total (t : TowelCount) : ℕ :=
  t.green + t.white + t.blue

/-- Represents the number of towels given away each day -/
structure DailyGiveaway where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the remaining towels after giving away for a number of days -/
def remainingTowels (initial : TowelCount) (daily : DailyGiveaway) (days : ℕ) : TowelCount :=
  { green := initial.green - daily.green * days,
    white := initial.white - daily.white * days,
    blue := initial.blue - daily.blue * days }

theorem maria_towel_problem :
  let initial := TowelCount.mk 35 21 15
  let daily := DailyGiveaway.mk 3 1 1
  let days := 7
  let remaining := remainingTowels initial daily days
  remaining.total = 36 := by sorry

end maria_towel_problem_l2475_247593


namespace quadratic_equations_solutions_l2475_247508

theorem quadratic_equations_solutions : 
  ∃ (s : Set ℝ), s = {0, 2, (6:ℝ)/5, -(6:ℝ)/5, -3, -7, 3, 1} ∧
  (∀ x ∈ s, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0) ∧
  (∀ x : ℝ, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0 → x ∈ s) :=
by sorry

end quadratic_equations_solutions_l2475_247508


namespace power_sum_of_i_l2475_247556

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^58 = -1 - i := by sorry

end power_sum_of_i_l2475_247556


namespace hemisphere_surface_area_l2475_247564

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 100 * π → 2 * π * r^2 + π * r^2 = 300 * π := by
  sorry

end hemisphere_surface_area_l2475_247564


namespace necessary_but_not_sufficient_l2475_247559

theorem necessary_but_not_sufficient :
  let p : ℝ → Prop := λ x ↦ |x + 1| > 2
  let q : ℝ → Prop := λ x ↦ x > 2
  (∀ x, ¬(q x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x) := by
  sorry

end necessary_but_not_sufficient_l2475_247559


namespace min_value_theorem_l2475_247511

theorem min_value_theorem (x y a b : ℝ) (h1 : x - 2*y - 2 ≤ 0) 
  (h2 : x + y - 2 ≤ 0) (h3 : 2*x - y + 2 ≥ 0) (ha : a > 0) (hb : b > 0)
  (h4 : ∀ (x' y' : ℝ), x' - 2*y' - 2 ≤ 0 → x' + y' - 2 ≤ 0 → 2*x' - y' + 2 ≥ 0 
    → a*x' + b*y' + 5 ≥ a*x + b*y + 5)
  (h5 : a*x + b*y + 5 = 2) :
  (2/a + 3/b : ℝ) ≥ (10 + 4*Real.sqrt 6)/3 :=
sorry

end min_value_theorem_l2475_247511


namespace min_value_theorem_l2475_247523

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  a + 4 * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ + 4 * b₀ = 9 :=
by sorry

end min_value_theorem_l2475_247523


namespace jacobs_gift_budget_l2475_247506

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 := by
  sorry

end jacobs_gift_budget_l2475_247506


namespace min_value_sum_min_value_sum_exact_l2475_247514

theorem min_value_sum (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y - (x + y) = 1 → a + b ≤ x + y :=
by sorry

theorem min_value_sum_exact (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  a + b = 2 * (Real.sqrt 2 + 1) :=
by sorry

end min_value_sum_min_value_sum_exact_l2475_247514


namespace function_value_at_three_l2475_247561

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem function_value_at_three
    (f : ℝ → ℝ)
    (h1 : FunctionalEquation f)
    (h2 : f 1 = 2) :
    f 3 = 12 := by
  sorry

end function_value_at_three_l2475_247561


namespace square_ending_theorem_l2475_247557

theorem square_ending_theorem (n : ℤ) :
  (∀ d : ℕ, d ∈ Finset.range 9 → (n^2 : ℤ) % 10000 ≠ d * 1111) ∧
  ((∃ d : ℕ, d ∈ Finset.range 9 ∧ (n^2 : ℤ) % 1000 = d * 111) → (n^2 : ℤ) % 1000 = 444) :=
by sorry

end square_ending_theorem_l2475_247557


namespace simplify_expression_l2475_247573

theorem simplify_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ b) :
  (1 - a) + (1 - b) = 2 := by sorry

end simplify_expression_l2475_247573


namespace product_of_powers_and_primes_l2475_247502

theorem product_of_powers_and_primes :
  2^4 * 3 * 5^3 * 7 * 11 = 2310000 := by
  sorry

end product_of_powers_and_primes_l2475_247502


namespace unique_n_l2475_247530

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (k : ℕ) (d : ℕ → ℕ+),
    k ≥ 6 ∧
    (∀ i ≤ k, d i ∣ n) ∧
    (∀ i j, i < j → d i < d j) ∧
    d 1 = 1 ∧
    d k = n ∧
    n = (d 5)^2 + (d 6)^2

theorem unique_n : ∀ n : ℕ+, is_valid_n n → n = 500 := by
  sorry

end unique_n_l2475_247530


namespace rectangular_program_box_indicates_input_output_l2475_247560

/-- Represents the function of a program box in an algorithm -/
inductive ProgramBoxFunction
  | StartEnd
  | InputOutput
  | AssignmentCalculation
  | ConnectBoxes

/-- The function of a rectangular program box in an algorithm -/
def rectangularProgramBoxFunction : ProgramBoxFunction := ProgramBoxFunction.InputOutput

/-- Theorem stating that a rectangular program box indicates input and output information -/
theorem rectangular_program_box_indicates_input_output :
  rectangularProgramBoxFunction = ProgramBoxFunction.InputOutput := by
  sorry

end rectangular_program_box_indicates_input_output_l2475_247560


namespace range_of_f_l2475_247538

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {0, 1, 2, 3} :=
by sorry

end range_of_f_l2475_247538


namespace rectangle_count_l2475_247548

/-- The number of rows and columns in the square grid -/
def gridSize : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different rectangles in a gridSize x gridSize square array of dots -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem rectangle_count : numRectangles = 100 := by
  sorry

end rectangle_count_l2475_247548


namespace smaller_number_is_24_l2475_247521

theorem smaller_number_is_24 (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) :
  min x y = 24 := by
sorry

end smaller_number_is_24_l2475_247521


namespace shopkeeper_theft_loss_l2475_247534

/-- Calculates the total percentage loss due to thefts for a shopkeeper --/
theorem shopkeeper_theft_loss (X : ℝ) (h : X > 0) : 
  let remaining_after_first_theft := 0.7 * X
  let remaining_after_first_sale := 0.75 * remaining_after_first_theft
  let remaining_after_second_theft := 0.6 * remaining_after_first_sale
  let remaining_after_second_sale := 0.7 * remaining_after_second_theft
  let final_remaining := 0.8 * remaining_after_second_sale
  (X - final_remaining) / X * 100 = 82.36 := by
sorry

end shopkeeper_theft_loss_l2475_247534


namespace paving_problem_l2475_247576

/-- Represents a worker paving paths in a park -/
structure Worker where
  speed : ℝ
  path_length : ℝ

/-- Represents the paving scenario in the park -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  total_time : ℝ

/-- The theorem statement for the paving problem -/
theorem paving_problem (scenario : PavingScenario) :
  scenario.worker1.speed > 0 ∧
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.total_time = 9 ∧
  scenario.worker1.path_length * scenario.worker1.speed = scenario.worker2.path_length * scenario.worker2.speed ∧
  scenario.worker2.path_length = scenario.worker1.path_length + 2 * (scenario.worker2.path_length / 12) →
  (scenario.worker2.path_length / 12) / scenario.worker2.speed * 60 = 45 := by
  sorry

#check paving_problem

end paving_problem_l2475_247576


namespace max_colors_without_monochromatic_trapezium_l2475_247542

/-- Regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Sequence of regular n-gons where each subsequent polygon's vertices are midpoints of the previous polygon's edges -/
def NGonSequence (n : ℕ) (m : ℕ) : ℕ → RegularNGon n
  | 0 => sorry
  | i + 1 => sorry

/-- A coloring of vertices of m n-gons using k colors -/
def Coloring (n m k : ℕ) := Fin m → Fin n → Fin k

/-- Predicate to check if four points form an isosceles trapezium -/
def IsIsoscelesTrapezium (a b c d : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a coloring contains a monochromatic isosceles trapezium -/
def HasMonochromaticIsoscelesTrapezium (n m k : ℕ) (coloring : Coloring n m k) : Prop := sorry

/-- The maximum number of colors that can be used without forming a monochromatic isosceles trapezium -/
theorem max_colors_without_monochromatic_trapezium 
  (n : ℕ) (m : ℕ) (h : m ≥ n^2 - n + 1) :
  (∃ (k : ℕ), k = n - 1 ∧ 
    (∀ (coloring : Coloring n m k), HasMonochromaticIsoscelesTrapezium n m k coloring) ∧
    (∃ (coloring : Coloring n m (k + 1)), ¬HasMonochromaticIsoscelesTrapezium n m (k + 1) coloring)) :=
sorry

end max_colors_without_monochromatic_trapezium_l2475_247542


namespace tire_usage_calculation_l2475_247504

/-- Calculates the miles each tire is used given the total number of tires, 
    simultaneously used tires, and total miles driven. -/
def miles_per_tire (total_tires : ℕ) (used_tires : ℕ) (total_miles : ℕ) : ℚ :=
  (total_miles * used_tires : ℚ) / total_tires

theorem tire_usage_calculation :
  let total_tires : ℕ := 6
  let used_tires : ℕ := 5
  let total_miles : ℕ := 42000
  miles_per_tire total_tires used_tires total_miles = 35000 := by
  sorry

end tire_usage_calculation_l2475_247504


namespace vector_sum_equality_implies_same_direction_l2475_247587

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def same_direction (a b : n) : Prop := ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_sum_equality_implies_same_direction (a b : n) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖a‖ + ‖b‖) :
  same_direction a b := by sorry

end vector_sum_equality_implies_same_direction_l2475_247587


namespace divisor_product_1024_implies_16_l2475_247592

/-- Given a positive integer n, returns the product of all its positive integer divisors. -/
def divisorProduct (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: If the product of the positive integer divisors of n is 1024, then n = 16. -/
theorem divisor_product_1024_implies_16 (n : ℕ+) :
  divisorProduct n = 1024 → n = 16 := by
  sorry

end divisor_product_1024_implies_16_l2475_247592


namespace square_area_given_edge_expressions_l2475_247513

theorem square_area_given_edge_expressions (x : ℚ) :
  (5 * x - 20 : ℚ) = (30 - 4 * x : ℚ) →
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 4900 / 81 := by
  sorry

end square_area_given_edge_expressions_l2475_247513


namespace dante_coconuts_left_l2475_247570

theorem dante_coconuts_left (paolo_coconuts : ℕ) (dante_coconuts : ℕ) (sold_coconuts : ℕ) : 
  paolo_coconuts = 14 →
  dante_coconuts = 3 * paolo_coconuts →
  sold_coconuts = 10 →
  dante_coconuts - sold_coconuts = 32 :=
by
  sorry

end dante_coconuts_left_l2475_247570


namespace star_polygon_external_intersection_angle_l2475_247505

/-- 
The angle at each intersection point outside a star-polygon with n points (n > 4) 
inscribed in a circle, given that each internal angle is (180(n-4))/n degrees.
-/
theorem star_polygon_external_intersection_angle (n : ℕ) (h : n > 4) : 
  let internal_angle := (180 * (n - 4)) / n
  (360 * (n - 4)) / n = 360 - 2 * (180 - internal_angle) := by
  sorry

#check star_polygon_external_intersection_angle

end star_polygon_external_intersection_angle_l2475_247505


namespace horner_method_v3_l2475_247577

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (x : ℝ) : ℝ :=
  ((((a * x + b) * x + c) * x + d) * x + e) * x + f

theorem horner_method_v3 : 
  horner_v3 3 (-2) 2 (-4) 0 (-7) 2 = 16 :=
by sorry

end horner_method_v3_l2475_247577


namespace unique_solution_condition_l2475_247522

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, a * x - 7 + (b + 2) * x = 3) ↔ a ≠ -b - 2 := by
  sorry

end unique_solution_condition_l2475_247522


namespace inequality_proof_l2475_247524

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * (b/a)^(1/3 : ℝ) + b * (c/b)^(1/3 : ℝ) + c * (a/c)^(1/3 : ℝ) ≤ a*b + b*c + c*a + 2/3 := by
  sorry

end inequality_proof_l2475_247524


namespace duck_pond_problem_l2475_247500

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) : 
  large_pond = 80 →
  (small_pond * 20 : ℕ) / 100 + (large_pond * 15 : ℕ) / 100 = ((small_pond + large_pond) * 16 : ℕ) / 100 →
  small_pond = 20 := by
sorry

end duck_pond_problem_l2475_247500


namespace continuity_at_6_l2475_247589

def f (x : ℝ) := 5 * x^2 - 1

theorem continuity_at_6 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by sorry

end continuity_at_6_l2475_247589


namespace log_sqrt2_and_inequality_l2475_247572

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt2_and_inequality :
  (log 4 (Real.sqrt 2) = 1/4) ∧
  (∀ x : ℝ, log x (Real.sqrt 2) > 1 ↔ 1 < x ∧ x < Real.sqrt 2) :=
by sorry

end log_sqrt2_and_inequality_l2475_247572


namespace milan_phone_bill_l2475_247598

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute. -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : minutes_billed 23.36 2 0.12 = 178 := by
  sorry

end milan_phone_bill_l2475_247598


namespace labourer_savings_l2475_247540

/-- Calculates the amount saved by a labourer after clearing debt -/
theorem labourer_savings (monthly_income : ℕ) (initial_expenditure : ℕ) (reduced_expenditure : ℕ) : 
  monthly_income = 78 → 
  initial_expenditure = 85 → 
  reduced_expenditure = 60 → 
  (4 * monthly_income - (4 * reduced_expenditure + (6 * initial_expenditure - 6 * monthly_income))) = 30 := by
sorry

end labourer_savings_l2475_247540


namespace torn_sheets_count_l2475_247535

/-- Represents a book with numbered pages -/
structure Book where
  /-- Each sheet contains two pages -/
  pages_per_sheet : Nat
  /-- The first torn-out page number -/
  first_torn_page : Nat
  /-- The last torn-out page number -/
  last_torn_page : Nat

/-- Check if two numbers have the same digits -/
def same_digits (a b : Nat) : Prop := sorry

/-- Calculate the number of torn-out sheets -/
def torn_sheets (book : Book) : Nat := sorry

/-- Main theorem -/
theorem torn_sheets_count (book : Book) :
  book.pages_per_sheet = 2 →
  book.first_torn_page = 185 →
  same_digits book.first_torn_page book.last_torn_page →
  Even book.last_torn_page →
  torn_sheets book = 167 := by
  sorry

end torn_sheets_count_l2475_247535


namespace power_function_m_value_l2475_247526

/-- A function f is a power function if it has the form f(x) = ax^n where a is a non-zero constant and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- Given that f(x) = (2m+3)x^(m^2-3) is a power function, prove that m = -1 -/
theorem power_function_m_value (m : ℝ) 
    (h : IsPowerFunction (fun x ↦ (2*m+3) * x^(m^2-3))) : 
  m = -1 := by
  sorry

end power_function_m_value_l2475_247526


namespace smallest_positive_integer_to_multiple_of_five_l2475_247546

theorem smallest_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (725 + m) % 5 = 0 → m ≥ n) ∧ (725 + n) % 5 = 0 :=
by sorry

end smallest_positive_integer_to_multiple_of_five_l2475_247546


namespace base5_product_theorem_l2475_247517

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Multiplies two base-5 numbers --/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_theorem :
  multiplyBase5 [1, 3, 2] [3, 1] = [3, 0, 0, 1, 4] := by sorry

end base5_product_theorem_l2475_247517


namespace imaginary_part_of_complex_fraction_l2475_247553

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i) / (1 + i)
  Complex.im z = 1 := by sorry

end imaginary_part_of_complex_fraction_l2475_247553


namespace local_extremum_properties_l2475_247527

/-- A function with a local extremum -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 1

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_properties (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f a b (-1) = 4) →
  (a = -3 ∧ b = -9 ∧
   ∀ x ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) x ≤ -1 ∧
   ∃ y ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) y = -28 ∧
   ∀ z ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) z ≥ -28) := by sorry

end local_extremum_properties_l2475_247527


namespace damien_picked_fraction_l2475_247586

/-- Proves that Damien picked 3/5 of the fruits from the trees --/
theorem damien_picked_fraction (apples plums : ℕ) (picked_fraction : ℚ) : 
  apples = 3 * plums →  -- The number of apples is three times the number of plums
  apples = 180 →  -- The initial number of apples is 180
  (1 - picked_fraction) * (apples + plums) = 96 →  -- After picking, 96 fruits remain
  picked_fraction = 3 / 5 := by
sorry


end damien_picked_fraction_l2475_247586


namespace homework_problem_count_l2475_247551

theorem homework_problem_count (p t : ℕ) (hp : p > 10) (ht : t > 2) : 
  p * t = (2 * p - 6) * (t - 2) → p * t = 96 := by
  sorry

end homework_problem_count_l2475_247551


namespace tan_alpha_value_l2475_247562

theorem tan_alpha_value (α : Real) 
  (h1 : π/2 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 10 / 5) : 
  Real.tan α = -3 := by
sorry

end tan_alpha_value_l2475_247562


namespace parallelepiped_volume_l2475_247599

/-- Given a rectangular parallelepiped with face areas p, q, and r, its volume is √(pqr) -/
theorem parallelepiped_volume (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (V : ℝ), V > 0 ∧ V * V = p * q * r :=
by sorry

end parallelepiped_volume_l2475_247599


namespace computer_operations_l2475_247547

/-- Represents the performance of a computer --/
structure ComputerPerformance where
  additions_per_second : ℕ
  multiplications_per_second : ℕ
  hours : ℕ

/-- Calculates the total number of operations a computer can perform --/
def total_operations (cp : ComputerPerformance) : ℕ :=
  (cp.additions_per_second + cp.multiplications_per_second) * (cp.hours * 3600)

/-- Theorem: A computer with given specifications performs 388,800,000 operations in 3 hours --/
theorem computer_operations :
  ∃ (cp : ComputerPerformance),
    cp.additions_per_second = 12000 ∧
    cp.multiplications_per_second = 2 * cp.additions_per_second ∧
    cp.hours = 3 ∧
    total_operations cp = 388800000 := by
  sorry


end computer_operations_l2475_247547


namespace imaginary_part_of_z_l2475_247590

theorem imaginary_part_of_z (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l2475_247590


namespace student_calculation_error_l2475_247568

/-- Represents a repeating decimal of the form 1.̅cd̅ where c and d are single digits -/
def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

/-- The difference between the correct calculation and the student's miscalculation -/
def calculation_difference (c d : ℕ) : ℚ :=
  84 * (repeating_decimal c d - (1 + (c : ℚ) / 10 + (d : ℚ) / 100))

theorem student_calculation_error :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ calculation_difference c d = 0.6 ∧ c * 10 + d = 71 := by
  sorry

end student_calculation_error_l2475_247568


namespace boat_payment_l2475_247520

theorem boat_payment (total : ℚ) (a b c d e : ℚ) : 
  total = 120 →
  a = (1/3) * (b + c + d + e) →
  b = (1/4) * (a + c + d + e) →
  c = (1/5) * (a + b + d + e) →
  d = 2 * e →
  a + b + c + d + e = total →
  e = 40/3 := by
sorry

end boat_payment_l2475_247520


namespace rationalize_and_sum_l2475_247541

theorem rationalize_and_sum : ∃ (A B C D E F : ℤ),
  (F > 0) ∧
  (∃ (k : ℚ), k ≠ 0 ∧ 
    k * (1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)) = 
    (A * Real.sqrt 5 + B * Real.sqrt 3 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
  (A + B + C + D + E + F = 196) := by
  sorry

end rationalize_and_sum_l2475_247541


namespace qt_length_in_specific_quadrilateral_l2475_247580

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: Length of QT in a specific quadrilateral -/
theorem qt_length_in_specific_quadrilateral 
  (PQRS : Quadrilateral) 
  (T : Point) 
  (h1 : distance PQRS.P PQRS.Q = 15)
  (h2 : distance PQRS.R PQRS.S = 20)
  (h3 : distance PQRS.P PQRS.R = 22)
  (h4 : triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) :
  distance PQRS.Q T = 66 / 7 := by
  sorry

end qt_length_in_specific_quadrilateral_l2475_247580


namespace water_in_first_tank_l2475_247512

theorem water_in_first_tank (capacity : ℝ) (water_second : ℝ) (fill_percentage : ℝ) (additional_water : ℝ) :
  capacity > 0 →
  water_second = 450 →
  fill_percentage = 0.45 →
  water_second = fill_percentage * capacity →
  additional_water = 1250 →
  additional_water + water_second + (capacity - water_second) = 2 * capacity →
  capacity - (additional_water + water_second) = 300 :=
by sorry

end water_in_first_tank_l2475_247512


namespace field_length_problem_l2475_247507

theorem field_length_problem (w l : ℝ) (h1 : l = 2 * w) (h2 : 36 = (1 / 8) * (l * w)) : l = 24 :=
by sorry

end field_length_problem_l2475_247507


namespace system_solution_l2475_247597

theorem system_solution (a : ℝ) :
  ∃ (x y z : ℝ),
    x^2 + y^2 - 2*z^2 = 2*a^2 ∧
    x + y + 2*z = 4*(a^2 + 1) ∧
    z^2 - x*y = a^2 ∧
    ((x = a^2 + a + 1 ∧ y = a^2 - a + 1) ∨ (x = a^2 - a + 1 ∧ y = a^2 + a + 1)) ∧
    z = a^2 + 1 := by
  sorry

end system_solution_l2475_247597


namespace max_positive_integers_l2475_247503

/-- A circular arrangement of 100 nonzero integers -/
def CircularArrangement := Fin 100 → ℤ

/-- Predicate to check if an arrangement satisfies the given condition -/
def SatisfiesCondition (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, arr i ≠ 0 ∧ arr i > arr ((i + 1) % 100) * arr ((i + 2) % 100)

/-- Count of positive integers in an arrangement -/
def PositiveCount (arr : CircularArrangement) : ℕ :=
  (Finset.univ.filter (fun i => arr i > 0)).card

/-- Theorem stating the maximum number of positive integers possible -/
theorem max_positive_integers (arr : CircularArrangement) 
  (h : SatisfiesCondition arr) : PositiveCount arr ≤ 50 := by
  sorry

end max_positive_integers_l2475_247503


namespace problem_solution_l2475_247516

theorem problem_solution (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.5) 
  (h3 : 0 ≤ p_A ∧ p_A ≤ 1) (h4 : 0 ≤ p_B ∧ p_B ≤ 1) :
  1 - (1 - p_A) * (1 - p_B) = 0.7 := by
  sorry

end problem_solution_l2475_247516


namespace mikes_score_l2475_247588

def passing_threshold : ℝ := 0.30
def max_score : ℕ := 770
def shortfall : ℕ := 19

theorem mikes_score : 
  ⌊(passing_threshold * max_score : ℝ)⌋ - shortfall = 212 := by
  sorry

end mikes_score_l2475_247588


namespace radical_simplification_l2475_247583

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt (10 * q) := by
  sorry

end radical_simplification_l2475_247583


namespace combined_large_cheese_volume_l2475_247501

/-- The volume of a normal rectangular block of cheese in cubic feet -/
def normal_rectangular_volume : ℝ := 4

/-- The volume of a normal cylindrical block of cheese in cubic feet -/
def normal_cylindrical_volume : ℝ := 6

/-- The width multiplier for a large rectangular block -/
def large_rect_width_mult : ℝ := 1.5

/-- The depth multiplier for a large rectangular block -/
def large_rect_depth_mult : ℝ := 3

/-- The length multiplier for a large rectangular block -/
def large_rect_length_mult : ℝ := 2

/-- The radius multiplier for a large cylindrical block -/
def large_cyl_radius_mult : ℝ := 2

/-- The height multiplier for a large cylindrical block -/
def large_cyl_height_mult : ℝ := 3

/-- Theorem stating the combined volume of a large rectangular block and a large cylindrical block -/
theorem combined_large_cheese_volume :
  (normal_rectangular_volume * large_rect_width_mult * large_rect_depth_mult * large_rect_length_mult) +
  (normal_cylindrical_volume * large_cyl_radius_mult^2 * large_cyl_height_mult) = 108 := by
  sorry

end combined_large_cheese_volume_l2475_247501


namespace atop_distributive_laws_l2475_247519

-- Define the @ operation
def atop (a b : ℝ) : ℝ := a + 2 * b

-- State the theorem
theorem atop_distributive_laws :
  (∀ x y z : ℝ, x * (atop y z) = atop (x * y) (x * z)) ∧
  (∃ x y z : ℝ, atop x (y * z) ≠ (atop x y) * (atop x z)) ∧
  (∃ x y z : ℝ, atop (atop x y) (atop x z) ≠ atop x (y * z)) := by
  sorry

end atop_distributive_laws_l2475_247519


namespace tangent_parallel_points_l2475_247529

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
sorry

end tangent_parallel_points_l2475_247529


namespace second_set_length_is_twenty_l2475_247518

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

theorem second_set_length_is_twenty : second_set_length = 20 := by
  sorry

end second_set_length_is_twenty_l2475_247518


namespace quadratic_no_roots_l2475_247558

/-- A quadratic function with no real roots has a coefficient greater than 1 -/
theorem quadratic_no_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end quadratic_no_roots_l2475_247558


namespace no_solutions_to_equation_l2475_247596

theorem no_solutions_to_equation :
  ∀ x : ℝ, x ≠ 0 → x ≠ 5 → (2 * x^2 - 10 * x) / (x^2 - 5 * x) ≠ x - 3 := by
  sorry

end no_solutions_to_equation_l2475_247596


namespace state_quarters_fraction_l2475_247537

theorem state_quarters_fraction :
  ∀ (total_quarters : ℕ) (states_in_decade : ℕ),
    total_quarters = 18 →
    states_in_decade = 5 →
    (states_in_decade : ℚ) / (total_quarters : ℚ) = 5 / 18 := by
  sorry

end state_quarters_fraction_l2475_247537


namespace function_inequality_solution_set_l2475_247543

open Set
open Function

theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x, deriv f x < (1/2)) :
  {x | f x < x/2 + 1/2} = {x | x > 1} := by
sorry

end function_inequality_solution_set_l2475_247543


namespace johns_children_l2475_247591

theorem johns_children (john_notebooks : ℕ → ℕ) (wife_notebooks : ℕ → ℕ) (total_notebooks : ℕ) :
  (∀ c : ℕ, john_notebooks c = 2 * c) →
  (∀ c : ℕ, wife_notebooks c = 5 * c) →
  (∃ c : ℕ, john_notebooks c + wife_notebooks c = total_notebooks) →
  total_notebooks = 21 →
  ∃ c : ℕ, c = 3 ∧ john_notebooks c + wife_notebooks c = total_notebooks :=
by sorry

end johns_children_l2475_247591


namespace correct_operation_l2475_247525

theorem correct_operation (m : ℝ) : 3 * m^2 * (2 * m^3) = 6 * m^5 := by
  sorry

end correct_operation_l2475_247525


namespace holiday_ticket_cost_theorem_l2475_247566

def holiday_ticket_cost (regular_adult_price : ℝ) : ℝ :=
  let holiday_adult_price := 1.1 * regular_adult_price
  let child_price := 0.5 * regular_adult_price
  6 * holiday_adult_price + 5 * child_price

theorem holiday_ticket_cost_theorem (regular_adult_price : ℝ) :
  4 * (1.1 * regular_adult_price) + 3 * (0.5 * regular_adult_price) = 28.80 →
  holiday_ticket_cost regular_adult_price = 44.41 := by
  sorry

end holiday_ticket_cost_theorem_l2475_247566


namespace triangle_problem_l2475_247552

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  2 * Real.sin A = 3 * Real.sin B →
  Real.cos B = 3 * Real.sqrt 10 / 10 →
  a = 3 ∧ b = 2 ∧ Real.sin (2 * A) = (3 - 4 * Real.sqrt 3) / 10 :=
by sorry

end triangle_problem_l2475_247552


namespace a_values_l2475_247584

def A (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def B : Set ℝ := {1, 3}

theorem a_values (a : ℝ) : (A a ∩ B = {1, 3}) → (a = -1 ∨ a = 4) := by
  sorry

end a_values_l2475_247584


namespace range_of_a_l2475_247571

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 ≤ 0 → x^2 - a*x - a - 2 ≤ 0) ↔ a ≥ 2/3 := by
  sorry

end range_of_a_l2475_247571


namespace sum_of_cubes_l2475_247545

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l2475_247545


namespace fraction_calculation_l2475_247531

theorem fraction_calculation : 
  (1 / 5 + 1 / 7) / (3 / 8 - 1 / 9) = 864 / 665 := by sorry

end fraction_calculation_l2475_247531


namespace freds_carrots_l2475_247554

/-- Given that Sally grew 6 carrots and the total number of carrots is 10,
    prove that Fred grew 4 carrots. -/
theorem freds_carrots (sally_carrots : ℕ) (total_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : total_carrots = 10) :
  total_carrots - sally_carrots = 4 := by
  sorry

end freds_carrots_l2475_247554


namespace line_mn_properties_l2475_247567

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for the sum of vertical coordinates
def sum_of_verticals (m n : ℝ × ℝ) : Prop := m.2 + n.2 = 4

-- Define the angle condition
def angle_condition (m n : ℝ × ℝ) : Prop :=
  (m.2 / (m.1 + 2)) + (n.2 / (n.1 + 2)) = 0

-- Main theorem
theorem line_mn_properties (m n : ℝ × ℝ) :
  parabola m → parabola n → sum_of_verticals m n → angle_condition m n →
  ∃ k b : ℝ, k = 1 ∧ b = -2 ∧ ∀ x y : ℝ, y = k * x + b ↔ (x = m.1 ∧ y = m.2) ∨ (x = n.1 ∧ y = n.2) :=
sorry

end line_mn_properties_l2475_247567


namespace different_pairs_eq_48_l2475_247565

/-- The number of distinct mystery novels -/
def mystery_novels : ℕ := 4

/-- The number of distinct fantasy novels -/
def fantasy_novels : ℕ := 4

/-- The number of distinct biographies -/
def biographies : ℕ := 4

/-- The number of genres -/
def num_genres : ℕ := 3

/-- The number of different pairs of books that can be chosen -/
def different_pairs : ℕ := num_genres * mystery_novels * fantasy_novels

theorem different_pairs_eq_48 : different_pairs = 48 := by
  sorry

end different_pairs_eq_48_l2475_247565


namespace water_experiment_proof_l2475_247532

/-- Calculates the remaining amount of water after an experiment -/
def remaining_water (initial : ℚ) (used : ℚ) : ℚ :=
  initial - used

/-- Proves that given 3 gallons of water and using 5/4 gallons, the remaining amount is 7/4 gallons -/
theorem water_experiment_proof :
  remaining_water 3 (5/4) = 7/4 := by
  sorry

end water_experiment_proof_l2475_247532


namespace quadratic_max_value_l2475_247582

theorem quadratic_max_value (m : ℝ) : 
  (∃ (y : ℝ → ℝ), 
    (∀ x, y x = -(x - m)^2 + m^2 + 1) ∧ 
    (∀ x, -2 ≤ x ∧ x ≤ 1 → y x ≤ 4) ∧
    (∃ x, -2 ≤ x ∧ x ≤ 1 ∧ y x = 4)) →
  m = 2 ∨ m = -Real.sqrt 3 := by
sorry

end quadratic_max_value_l2475_247582


namespace expression_value_l2475_247533

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 2) : 
  2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0 :=
by sorry

end expression_value_l2475_247533


namespace optimal_speed_theorem_l2475_247544

theorem optimal_speed_theorem (d t : ℝ) 
  (h1 : d = 45 * (t + 1/15))
  (h2 : d = 75 * (t - 1/15)) :
  d / t = 56.25 := by
  sorry

end optimal_speed_theorem_l2475_247544


namespace equation_solution_l2475_247539

theorem equation_solution (a b : ℝ) (h1 : 3 = (a + 5).sqrt) (h2 : 3 = (7 * a - 2 * b + 1)^(1/3)) :
  ∃ x : ℝ, (a * (x - 2)^2 - 9 * b = 0) ∧ (x = 7/2 ∨ x = 1/2) :=
by sorry

end equation_solution_l2475_247539


namespace analysis_method_seeks_sufficient_conditions_l2475_247528

/-- The analysis method in mathematical proofs --/
structure AnalysisMethod where
  conclusion : Prop
  seek_conditions : Prop → Prop

/-- Definition of sufficient conditions --/
def sufficient_conditions (am : AnalysisMethod) (conditions : Prop) : Prop :=
  conditions → am.conclusion

/-- Theorem stating that the analysis method seeks sufficient conditions --/
theorem analysis_method_seeks_sufficient_conditions (am : AnalysisMethod) :
  ∃ (conditions : Prop), sufficient_conditions am conditions :=
sorry

end analysis_method_seeks_sufficient_conditions_l2475_247528


namespace milo_run_distance_milo_two_hour_run_l2475_247595

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard speed in miles per hour -/
def milo_skateboard_speed : ℝ := 2 * milo_run_speed

/-- Cory's wheelchair speed in miles per hour -/
def cory_wheelchair_speed : ℝ := 12

theorem milo_run_distance : ℝ → ℝ
  | hours => milo_run_speed * hours

theorem milo_two_hour_run : milo_run_distance 2 = 6 := by
  sorry

end milo_run_distance_milo_two_hour_run_l2475_247595


namespace lines_1_and_4_are_perpendicular_l2475_247563

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope4 : ℚ := -4 / 3

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_1_and_4_are_perpendicular : 
  are_perpendicular slope1 slope4 := by sorry

end lines_1_and_4_are_perpendicular_l2475_247563


namespace square_of_99_l2475_247585

theorem square_of_99 : 99 ^ 2 = 9801 := by
  sorry

end square_of_99_l2475_247585


namespace algebraic_expression_value_l2475_247594

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 2 * x^2 + 3 * x - 7 = -6 := by
sorry

end algebraic_expression_value_l2475_247594


namespace weight_vest_savings_l2475_247510

theorem weight_vest_savings (weight_vest_cost plate_weight plate_cost_per_pound
                             weight_vest_200_cost weight_vest_200_discount : ℕ) :
  weight_vest_cost = 250 →
  plate_weight = 200 →
  plate_cost_per_pound = 12 / 10 →
  weight_vest_200_cost = 700 →
  weight_vest_200_discount = 100 →
  (weight_vest_200_cost - weight_vest_200_discount) - 
  (weight_vest_cost + plate_weight * plate_cost_per_pound) = 110 := by
sorry

end weight_vest_savings_l2475_247510


namespace crypto_deg_is_69_l2475_247515

/-- Represents the digits in the cryptographer's encoding -/
inductive CryptoDigit
| A | B | C | D | E | F | G

/-- Converts a CryptoDigit to its corresponding base 7 value -/
def cryptoToBase7 : CryptoDigit → Fin 7
| CryptoDigit.A => 0
| CryptoDigit.B => 1
| CryptoDigit.D => 3
| CryptoDigit.E => 2
| CryptoDigit.F => 5
| CryptoDigit.G => 6
| _ => 0  -- C is not used in this problem, so we assign it 0

/-- Represents a three-digit number in the cryptographer's encoding -/
structure CryptoNumber where
  hundreds : CryptoDigit
  tens : CryptoDigit
  ones : CryptoDigit

/-- Converts a CryptoNumber to its base 10 value -/
def cryptoToBase10 (n : CryptoNumber) : Nat :=
  (cryptoToBase7 n.hundreds).val * 49 +
  (cryptoToBase7 n.tens).val * 7 +
  (cryptoToBase7 n.ones).val

/-- The main theorem to prove -/
theorem crypto_deg_is_69 :
  let deg : CryptoNumber := ⟨CryptoDigit.D, CryptoDigit.E, CryptoDigit.G⟩
  cryptoToBase10 deg = 69 := by sorry

end crypto_deg_is_69_l2475_247515


namespace point_transformation_theorem_l2475_247536

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (h, k) := center
  let (x, y) := point
  (h - (y - k), k + (x - h))

def reflectAboutYEqualsX (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (y, x)

-- State the theorem
theorem point_transformation_theorem (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 3) P
  let final := reflectAboutYEqualsX rotated
  final = (4, -5) → b - a = -5 :=
by
  sorry

end point_transformation_theorem_l2475_247536


namespace complex_equation_solution_l2475_247578

theorem complex_equation_solution (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.I = (2 - Complex.I) / (1 + 2 * Complex.I) → b = -2 := by sorry

end complex_equation_solution_l2475_247578
