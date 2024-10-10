import Mathlib

namespace function_composition_equality_l727_72774

/-- Given a function f and a condition on f[g(x)], prove the form of g(x) -/
theorem function_composition_equality 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * x - 1) 
  (h_fg : ∀ x, f (g x) = 2 * x + 3) : 
  ∀ x, g x = (2/3) * x + (4/3) := by
sorry

end function_composition_equality_l727_72774


namespace q_function_equality_l727_72788

/-- Given a function q(x) that satisfies the equation
    q(x) + (2x^6 + 5x^4 + 10x) = (8x^4 + 35x^3 + 40x^2 + 2),
    prove that q(x) = -2x^6 + 3x^4 + 35x^3 + 40x^2 - 10x + 2 -/
theorem q_function_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x) = (8 * x^4 + 35 * x^3 + 40 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + 3 * x^4 + 35 * x^3 + 40 * x^2 - 10 * x + 2) :=
by sorry

end q_function_equality_l727_72788


namespace race_distance_l727_72757

theorem race_distance (a_time b_time lead_distance : ℕ) 
  (ha : a_time = 28)
  (hb : b_time = 32)
  (hl : lead_distance = 28) : 
  (b_time * lead_distance) / (b_time - a_time) = 224 := by
  sorry

end race_distance_l727_72757


namespace glasses_fraction_after_tripling_l727_72792

theorem glasses_fraction_after_tripling (n : ℝ) (h : n > 0) :
  let initial_with_glasses := (2 / 3 : ℝ) * n
  let initial_without_glasses := (1 / 3 : ℝ) * n
  let new_without_glasses := 3 * initial_without_glasses
  let new_total := initial_with_glasses + new_without_glasses
  initial_with_glasses / new_total = 2 / 5 := by
sorry

end glasses_fraction_after_tripling_l727_72792


namespace third_term_is_five_l727_72790

/-- An arithmetic sequence where the sum of the first and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 4 * d) = 10

/-- The third term of the arithmetic sequence -/
def ThirdTerm (a : ℝ) (d : ℝ) : ℝ :=
  a + 2 * d

theorem third_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) :
  ThirdTerm a d = 5 := by
  sorry


end third_term_is_five_l727_72790


namespace problem_solution_l727_72733

theorem problem_solution (n k : ℕ) : 
  (1/2)^n * (1/81)^k = 1/18^22 → k = 11 → n = 22 := by
  sorry

end problem_solution_l727_72733


namespace ethyne_bond_count_l727_72732

/-- Represents a chemical bond in a molecule -/
inductive Bond
  | Sigma
  | Pi

/-- Represents the ethyne (acetylene) molecule -/
structure Ethyne where
  /-- The number of carbon atoms in ethyne -/
  carbon_count : Nat
  /-- The number of hydrogen atoms in ethyne -/
  hydrogen_count : Nat
  /-- The structure of ethyne is linear -/
  is_linear : Bool
  /-- Each carbon atom forms a triple bond with the other carbon atom -/
  has_carbon_triple_bond : Bool
  /-- Each carbon atom forms a single bond with a hydrogen atom -/
  has_carbon_hydrogen_single_bond : Bool

/-- Counts the number of sigma bonds in ethyne -/
def count_sigma_bonds (e : Ethyne) : Nat :=
  sorry

/-- Counts the number of pi bonds in ethyne -/
def count_pi_bonds (e : Ethyne) : Nat :=
  sorry

/-- Theorem stating the number of sigma and pi bonds in ethyne -/
theorem ethyne_bond_count (e : Ethyne) :
  e.carbon_count = 2 ∧
  e.hydrogen_count = 2 ∧
  e.is_linear ∧
  e.has_carbon_triple_bond ∧
  e.has_carbon_hydrogen_single_bond →
  count_sigma_bonds e = 3 ∧ count_pi_bonds e = 2 :=
by sorry

end ethyne_bond_count_l727_72732


namespace negation_equivalence_l727_72713

variable (m : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - m*x - m < 0) ↔ (∀ x : ℝ, x^2 - m*x - m ≥ 0) :=
by sorry

end negation_equivalence_l727_72713


namespace train_crossing_time_l727_72755

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length_m : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length_m = 1250 →
  train_speed_kmh = 300 →
  crossing_time_s = 15 →
  crossing_time_s = (train_length_m / 1000) / (train_speed_kmh / 3600) := by
  sorry

#check train_crossing_time

end train_crossing_time_l727_72755


namespace inverse_variation_problem_l727_72799

-- Define the inverse variation relationship
def inverse_variation (a b k : ℝ) : Prop := a * b^3 = k

-- State the theorem
theorem inverse_variation_problem :
  ∀ (a₁ a₂ k : ℝ),
  inverse_variation a₁ 2 k →
  a₁ = 16 →
  inverse_variation a₂ 4 k →
  a₂ = 2 := by
sorry

end inverse_variation_problem_l727_72799


namespace sum_of_squares_theorem_l727_72719

theorem sum_of_squares_theorem (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_power : x^4 + y^4 + z^4 = x^6 + y^6 + z^6) :
  x^2 + y^2 + z^2 = 3/2 := by
sorry

end sum_of_squares_theorem_l727_72719


namespace log_base_values_l727_72728

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_base_values (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 2 4, f a x ∈ Set.Icc (f a 2) (f a 4)) ∧
  (f a 4 - f a 2 = 2 ∨ f a 2 - f a 4 = 2) →
  a = Real.sqrt 2 ∨ a = Real.sqrt 2 / 2 := by
sorry

end log_base_values_l727_72728


namespace f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l727_72772

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the functions
def f (x : ℝ) : ℝ := x^4 + x^2
def g (x : ℝ) : ℝ := x^3 + x^2

-- Theorem statements
theorem f_is_even : IsEven f := by sorry

theorem g_is_not_odd : ¬ IsOdd g := by sorry

theorem even_function_symmetry (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x y, f x = y ↔ f (-x) = y := by sorry

theorem odd_function_symmetry (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x y, f x = y ↔ f (-x) = -y := by sorry

end f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l727_72772


namespace charity_ticket_revenue_l727_72750

theorem charity_ticket_revenue :
  ∀ (f d : ℕ) (p : ℚ),
    f + d = 160 →
    f * p + d * (2/3 * p) = 2800 →
    ∃ (full_revenue : ℚ),
      full_revenue = f * p ∧
      full_revenue = 1680 :=
by sorry

end charity_ticket_revenue_l727_72750


namespace jacket_purchase_price_l727_72720

theorem jacket_purchase_price 
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (gross_profit : ℝ)
  (h_markup : markup_rate = 0.4)
  (h_discount : discount_rate = 0.2)
  (h_profit : gross_profit = 16) :
  ∃ (purchase_price selling_price : ℝ),
    selling_price = purchase_price + markup_rate * selling_price ∧
    gross_profit = (1 - discount_rate) * selling_price - purchase_price ∧
    purchase_price = 48 := by
  sorry

end jacket_purchase_price_l727_72720


namespace ninety_degrees_possible_l727_72748

-- Define a pentagon with angles in arithmetic progression
def Pentagon (a d : ℝ) : Prop :=
  a > 60 ∧  -- smallest angle > 60 degrees
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540  -- sum of angles in pentagon

-- Theorem statement
theorem ninety_degrees_possible (a d : ℝ) :
  Pentagon a d → ∃ k : ℕ, k < 5 ∧ a + k*d = 90 := by
  sorry


end ninety_degrees_possible_l727_72748


namespace composition_result_l727_72793

/-- Given two functions f and g, prove that f(g(-2)) = 81 -/
theorem composition_result :
  (f : ℝ → ℝ) →
  (g : ℝ → ℝ) →
  (∀ x, f x = x^2) →
  (∀ x, g x = 2*x - 5) →
  f (g (-2)) = 81 := by
sorry

end composition_result_l727_72793


namespace minimal_reciprocal_sum_l727_72722

def satisfies_equation (a b : ℕ+) : Prop := 30 - a.val = 4 * b.val

def reciprocal_sum (a b : ℕ+) : ℚ := 1 / a.val + 1 / b.val

theorem minimal_reciprocal_sum :
  ∀ a b : ℕ+, satisfies_equation a b →
    reciprocal_sum a b ≥ reciprocal_sum 10 5 :=
by sorry

end minimal_reciprocal_sum_l727_72722


namespace boat_speed_l727_72745

/-- The average speed of a boat in still water, given travel times with and against a current. -/
theorem boat_speed (time_with_current time_against_current current_speed : ℝ)
  (h1 : time_with_current = 2)
  (h2 : time_against_current = 2.5)
  (h3 : current_speed = 3)
  (h4 : time_with_current * (x + current_speed) = time_against_current * (x - current_speed)) :
  x = 27 :=
by sorry


end boat_speed_l727_72745


namespace book_chapters_l727_72706

/-- Represents the number of pages in a book with arithmetic progression of chapter lengths -/
def book_pages (n : ℕ) : ℕ := n * (2 * 13 + (n - 1) * 3) / 2

/-- Theorem stating that a book with 95 pages, where the first chapter has 13 pages
    and each subsequent chapter has 3 more pages than the previous one, has 5 chapters -/
theorem book_chapters :
  ∃ (n : ℕ), n > 0 ∧ book_pages n = 95 ∧ n = 5 := by sorry

end book_chapters_l727_72706


namespace prime_between_squares_l727_72734

theorem prime_between_squares : ∃ p : ℕ, 
  Prime p ∧ 
  (∃ n : ℕ, p = n^2 + 9) ∧ 
  (∃ m : ℕ, p = (m+1)^2 - 8) ∧ 
  p = 73 := by
sorry

end prime_between_squares_l727_72734


namespace solve_for_x_l727_72739

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end solve_for_x_l727_72739


namespace speed_increase_from_weight_cut_l727_72727

/-- Proves that the speed increase from weight cut is 10 mph given the initial conditions --/
theorem speed_increase_from_weight_cut 
  (original_speed : ℝ) 
  (supercharge_increase_percent : ℝ)
  (final_speed : ℝ) :
  original_speed = 150 →
  supercharge_increase_percent = 30 →
  final_speed = 205 →
  final_speed - (original_speed * (1 + supercharge_increase_percent / 100)) = 10 := by
sorry

end speed_increase_from_weight_cut_l727_72727


namespace gcd_problem_l727_72715

theorem gcd_problem (b : ℤ) (h : 1632 ∣ b) :
  Int.gcd (b^2 + 11*b + 30) (b + 6) = 6 := by
  sorry

end gcd_problem_l727_72715


namespace arithmetic_progression_properties_l727_72776

/-- An arithmetic progression with the property that the sum of its first n terms is 5n² for any n -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a₁ : ℝ
  /-- The common difference of the progression -/
  d : ℝ
  /-- Property: The sum of the first n terms is 5n² for any n -/
  sum_property : ∀ n : ℕ, n * (2 * a₁ + (n - 1) * d) / 2 = 5 * n^2

/-- Theorem stating the properties of the arithmetic progression -/
theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  ap.d = 10 ∧ ap.a₁ = 5 ∧ ap.a₁ + ap.d = 15 ∧ ap.a₁ + 2 * ap.d = 25 := by
  sorry

end arithmetic_progression_properties_l727_72776


namespace peanut_cluster_probability_theorem_l727_72771

/-- Represents the composition of a box of chocolates -/
structure ChocolateBox where
  total : Nat
  caramels : Nat
  nougats : Nat
  truffles : Nat
  peanut_clusters : Nat

/-- Calculates the probability of selecting a peanut cluster -/
def peanut_cluster_probability (box : ChocolateBox) : Rat :=
  box.peanut_clusters / box.total

/-- Theorem stating the probability of selecting a peanut cluster -/
theorem peanut_cluster_probability_theorem (box : ChocolateBox) 
  (h1 : box.total = 50)
  (h2 : box.caramels = 3)
  (h3 : box.nougats = 2 * box.caramels)
  (h4 : box.truffles = box.caramels + 6)
  (h5 : box.peanut_clusters = box.total - box.caramels - box.nougats - box.truffles) :
  peanut_cluster_probability box = 32 / 50 := by
  sorry

#eval (32 : Rat) / 50

end peanut_cluster_probability_theorem_l727_72771


namespace largest_number_from_hcf_lcm_factors_l727_72784

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 13 * 14) :
  max a b = 322 := by
  sorry

end largest_number_from_hcf_lcm_factors_l727_72784


namespace liquid_X_percentage_l727_72767

/-- The percentage of liquid X in solution P -/
def percentage_X_in_P : ℝ := sorry

/-- The percentage of liquid X in solution Q -/
def percentage_X_in_Q : ℝ := 0.015

/-- The weight of solution P in grams -/
def weight_P : ℝ := 200

/-- The weight of solution Q in grams -/
def weight_Q : ℝ := 800

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.013

theorem liquid_X_percentage :
  percentage_X_in_P * weight_P + percentage_X_in_Q * weight_Q =
  percentage_X_in_mixture * (weight_P + weight_Q) ∧
  percentage_X_in_P = 0.005 := by sorry

end liquid_X_percentage_l727_72767


namespace max_value_bound_max_value_achievable_l727_72753

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def max_value (a b c : V) : ℝ :=
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2

theorem max_value_bound (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (hc : ‖c‖ = 2) :
  max_value a b c ≤ 253 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : V), ‖a‖ = 3 ∧ ‖b‖ = 4 ∧ ‖c‖ = 2 ∧ max_value a b c = 253 :=
sorry

end max_value_bound_max_value_achievable_l727_72753


namespace negation_of_all_squares_positive_l727_72798

theorem negation_of_all_squares_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end negation_of_all_squares_positive_l727_72798


namespace two_visits_count_l727_72795

/-- Represents a friend's visiting pattern -/
structure VisitPattern where
  period : Nat
  offset : Nat

/-- Calculates the number of days where exactly two friends visit -/
def exactTwoVisits (alice beatrix claire : VisitPattern) (totalDays : Nat) : Nat :=
  sorry

theorem two_visits_count :
  let alice : VisitPattern := { period := 2, offset := 0 }
  let beatrix : VisitPattern := { period := 6, offset := 1 }
  let claire : VisitPattern := { period := 5, offset := 1 }
  let totalDays : Nat := 400
  exactTwoVisits alice beatrix claire totalDays = 80 := by sorry

end two_visits_count_l727_72795


namespace parallel_vectors_x_value_l727_72764

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 := by
    sorry

end parallel_vectors_x_value_l727_72764


namespace arithmetic_sequence_count_l727_72736

theorem arithmetic_sequence_count :
  ∀ (a l d : ℝ) (n : ℕ),
    a = 2.5 →
    l = 62.5 →
    d = 5 →
    l = a + (n - 1) * d →
    n = 13 :=
by sorry

end arithmetic_sequence_count_l727_72736


namespace incandescent_bulbs_on_l727_72743

/-- Prove that the number of switched-on incandescent bulbs is 420 -/
theorem incandescent_bulbs_on (total_bulbs : ℕ) 
  (incandescent_percent fluorescent_percent led_percent halogen_percent : ℚ)
  (total_on_percent : ℚ)
  (incandescent_on_percent fluorescent_on_percent led_on_percent halogen_on_percent : ℚ) :
  total_bulbs = 3000 →
  incandescent_percent = 40 / 100 →
  fluorescent_percent = 30 / 100 →
  led_percent = 20 / 100 →
  halogen_percent = 10 / 100 →
  total_on_percent = 55 / 100 →
  incandescent_on_percent = 35 / 100 →
  fluorescent_on_percent = 50 / 100 →
  led_on_percent = 80 / 100 →
  halogen_on_percent = 30 / 100 →
  (incandescent_percent * total_bulbs : ℚ) * incandescent_on_percent = 420 :=
by
  sorry


end incandescent_bulbs_on_l727_72743


namespace unbounded_fraction_over_primes_l727_72700

-- Define the ord_p function
def ord_p (a p : ℕ) : ℕ := sorry

-- State the theorem
theorem unbounded_fraction_over_primes (a : ℕ) (h : a > 1) :
  ∀ M : ℕ, ∃ p : ℕ, Prime p ∧ (p - 1) / ord_p a p > M :=
sorry

end unbounded_fraction_over_primes_l727_72700


namespace positive_root_of_cubic_l727_72721

theorem positive_root_of_cubic (x : ℝ) : 
  x = 2 + Real.sqrt 3 → x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 := by
  sorry

end positive_root_of_cubic_l727_72721


namespace minor_premise_incorrect_l727_72742

theorem minor_premise_incorrect : ¬ ∀ x : ℝ, x + 1/x ≥ 2 * Real.sqrt (x * (1/x)) := by
  sorry

end minor_premise_incorrect_l727_72742


namespace bisection_method_representation_l727_72716

/-- Represents different types of diagrams --/
inductive DiagramType
  | OrganizationalStructure
  | ProcessFlowchart
  | KnowledgeStructure
  | ProgramFlowchart

/-- Represents the bisection method algorithm --/
structure BisectionMethod where
  hasLoopStructure : Bool
  hasConditionalStructure : Bool

/-- Theorem stating that the bisection method for solving x^2 - 2 = 0 is best represented by a program flowchart --/
theorem bisection_method_representation (bm : BisectionMethod) 
  (h1 : bm.hasLoopStructure = true) 
  (h2 : bm.hasConditionalStructure = true) : 
  DiagramType.ProgramFlowchart = 
    (fun (d : DiagramType) => 
      if bm.hasLoopStructure ∧ bm.hasConditionalStructure 
      then DiagramType.ProgramFlowchart 
      else d) DiagramType.ProgramFlowchart :=
by
  sorry

#check bisection_method_representation

end bisection_method_representation_l727_72716


namespace harry_cookies_per_batch_l727_72726

/-- Calculates the number of cookies in a batch given the total chips, number of batches, and chips per cookie. -/
def cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  (total_chips / num_batches) / chips_per_cookie

/-- Proves that the number of cookies in a batch is 3 given the specified conditions. -/
theorem harry_cookies_per_batch :
  cookies_per_batch 81 3 9 = 3 := by
  sorry

end harry_cookies_per_batch_l727_72726


namespace prime_counting_upper_bound_l727_72761

open Real

/-- The prime counting function π(n) -/
noncomputable def prime_counting (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n > 55, π(n) < 3 ln 2 * (n / ln n) -/
theorem prime_counting_upper_bound (n : ℕ) (h : n > 55) :
  (prime_counting n : ℝ) < 3 * log 2 * (n / log n) := by
  sorry

end prime_counting_upper_bound_l727_72761


namespace reginalds_apple_sales_l727_72785

/-- Represents the problem of calculating the number of apples sold by Reginald --/
theorem reginalds_apple_sales :
  let apple_price : ℚ := 5 / 4  -- $1.25 per apple
  let bike_cost : ℚ := 80
  let repair_cost_ratio : ℚ := 1 / 4  -- 25% of bike cost
  let remaining_ratio : ℚ := 1 / 5  -- 1/5 of earnings remain after repairs
  let apples_per_set : ℕ := 6  -- 5 paid + 1 free
  let paid_apples_per_set : ℕ := 5

  ∃ (total_apples : ℕ),
    total_apples = 120 ∧
    total_apples % apples_per_set = 0 ∧
    let total_sets := total_apples / apples_per_set
    let total_earnings := (total_sets * paid_apples_per_set : ℚ) * apple_price
    let repair_cost := bike_cost * repair_cost_ratio
    total_earnings * remaining_ratio = total_earnings - repair_cost :=
by
  sorry

end reginalds_apple_sales_l727_72785


namespace queue_waiting_times_l727_72703

/-- Represents a queue with Slowpokes and Quickies -/
structure Queue where
  m : ℕ  -- number of Slowpokes
  n : ℕ  -- number of Quickies
  a : ℕ  -- time taken by Quickies
  b : ℕ  -- time taken by Slowpokes

/-- Calculates the minimum total waiting time for a given queue -/
def min_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.a * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the maximum total waiting time for a given queue -/
def max_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.b * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the expected total waiting time for a given queue -/
def expected_waiting_time (q : Queue) : ℚ :=
  (q.m + q.n).choose 2 * (q.b * q.m + q.a * q.n) / (q.m + q.n)

/-- Theorem stating the properties of the queue waiting times -/
theorem queue_waiting_times (q : Queue) :
  (min_waiting_time q ≤ max_waiting_time q) ∧
  (↑(min_waiting_time q) ≤ expected_waiting_time q) ∧
  (expected_waiting_time q ≤ max_waiting_time q) :=
sorry

end queue_waiting_times_l727_72703


namespace impossibility_of_zero_sum_l727_72783

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a configuration of signs between numbers 1 to 10 -/
def SignConfiguration := Fin 9 → Bool

/-- Calculates the sum based on a given sign configuration -/
def calculate_sum (config : SignConfiguration) : ℤ :=
  sorry

theorem impossibility_of_zero_sum : ∀ (config : SignConfiguration), 
  calculate_sum config ≠ 0 := by
  sorry

end impossibility_of_zero_sum_l727_72783


namespace max_product_l727_72768

def digits : Finset Nat := {3, 5, 6, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  {a, b, c, d, e} = digits ∧ 
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 ∧
  10 ≤ 10 * d + e ∧ 10 * d + e < 100

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat, is_valid_pair a b c d e →
    product a b c d e ≤ product 9 5 3 8 6 :=
by sorry

end max_product_l727_72768


namespace sum_of_complex_equation_l727_72763

/-- Given real numbers x and y satisfying (2+i)x = 4+yi, prove that x + y = 4 -/
theorem sum_of_complex_equation (x y : ℝ) : 
  (Complex.I : ℂ) * x + 2 * x = 4 + (Complex.I : ℂ) * y → x + y = 4 := by
  sorry

end sum_of_complex_equation_l727_72763


namespace total_cards_l727_72705

-- Define the number of people
def num_people : ℕ := 4

-- Define the number of cards each person has
def cards_per_person : ℕ := 14

-- Theorem: The total number of cards is 56
theorem total_cards : num_people * cards_per_person = 56 := by
  sorry

end total_cards_l727_72705


namespace store_sales_l727_72711

-- Define the prices and quantities of each pencil type
def eraser_price : ℚ := 0.8
def regular_price : ℚ := 0.5
def short_price : ℚ := 0.4
def mechanical_price : ℚ := 1.2
def novelty_price : ℚ := 1.5

def eraser_quantity : ℕ := 200
def regular_quantity : ℕ := 40
def short_quantity : ℕ := 35
def mechanical_quantity : ℕ := 25
def novelty_quantity : ℕ := 15

-- Define the total sales function
def total_sales : ℚ :=
  eraser_price * eraser_quantity +
  regular_price * regular_quantity +
  short_price * short_quantity +
  mechanical_price * mechanical_quantity +
  novelty_price * novelty_quantity

-- Theorem statement
theorem store_sales : total_sales = 246.5 := by
  sorry

end store_sales_l727_72711


namespace heartsuit_five_three_l727_72724

def heartsuit (x y : ℤ) : ℤ := 4 * x - 2 * y

theorem heartsuit_five_three : heartsuit 5 3 = 14 := by
  sorry

end heartsuit_five_three_l727_72724


namespace range_of_a_l727_72712

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → a ∈ Set.Iic (-1 : ℝ) :=
by sorry

end range_of_a_l727_72712


namespace correlated_relationships_l727_72749

/-- Represents a relationship between two variables -/
structure Relationship where
  has_correlation : Bool

/-- The relationship between carbon content in molten steel and smelting time -/
def steel_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a point on a curve and its coordinates -/
def curve_point_relationship : Relationship :=
  ⟨false⟩

/-- The relationship between citrus yield and temperature -/
def citrus_yield_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between tree cross-section diameter and height -/
def tree_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a person's age and wealth -/
def age_wealth_relationship : Relationship :=
  ⟨true⟩

/-- The list of all relationships -/
def all_relationships : List Relationship :=
  [steel_relationship, curve_point_relationship, citrus_yield_relationship, tree_relationship, age_wealth_relationship]

theorem correlated_relationships :
  (all_relationships.filter (·.has_correlation)).length = 4 :=
sorry

end correlated_relationships_l727_72749


namespace water_in_bucket_l727_72737

theorem water_in_bucket (initial_amount : ℝ) (poured_out : ℝ) : 
  initial_amount = 0.8 → poured_out = 0.2 → initial_amount - poured_out = 0.6 := by
  sorry

end water_in_bucket_l727_72737


namespace select_parents_count_l727_72725

/-- The number of ways to select 4 parents out of 12 (6 couples), 
    such that exactly one pair of the chosen 4 are a couple -/
def selectParents : ℕ := sorry

/-- The total number of couples -/
def totalCouples : ℕ := 6

/-- The total number of parents -/
def totalParents : ℕ := 12

/-- The number of parents to be selected -/
def parentsToSelect : ℕ := 4

theorem select_parents_count : 
  selectParents = 240 := by sorry

end select_parents_count_l727_72725


namespace constant_term_expansion_constant_term_is_21_l727_72718

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = x^7 + 2*x^6 + 2*x^5 + 3*x^4 + x^5 + 2*x^4 + x^3 + 7*x^3 + 7*x^2 + 21 := by
  sorry

theorem constant_term_is_21 : 
  (λ x : ℝ => (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) 0 = 21 := by
  sorry

end constant_term_expansion_constant_term_is_21_l727_72718


namespace power_function_range_l727_72702

def power_function (x : ℝ) (m : ℕ+) : ℝ := x^(3*m.val - 9)

theorem power_function_range (m : ℕ+) 
  (h1 : ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), y > x → power_function y m < power_function x m)
  (h2 : ∀ (x : ℝ), power_function x m = power_function (-x) m) :
  {a : ℝ | (a + 1)^(m.val/3) < (3 - 2*a)^(m.val/3)} = {a : ℝ | a < 2/3} := by
sorry

end power_function_range_l727_72702


namespace hostel_expenditure_hostel_expenditure_proof_l727_72738

/-- Calculates the new total expenditure of a hostel after adding more students --/
theorem hostel_expenditure 
  (initial_students : ℕ) 
  (budget_decrease : ℚ) 
  (expenditure_increase : ℚ) 
  (new_students : ℕ) : ℚ :=
  let new_total_students := initial_students + new_students
  let total_expenditure_increase := 
    (new_total_students : ℚ) * (initial_students : ℚ) * budget_decrease / initial_students + expenditure_increase
  total_expenditure_increase

/-- Proves that the new total expenditure is 5775 given the initial conditions --/
theorem hostel_expenditure_proof :
  hostel_expenditure 100 10 400 32 = 5775 := by
  sorry

end hostel_expenditure_hostel_expenditure_proof_l727_72738


namespace x_sixth_power_equals_one_l727_72746

theorem x_sixth_power_equals_one (x : ℝ) (h : 1 + x + x^2 + x^3 + x^4 + x^5 = 0) : x^6 = 1 := by
  sorry

end x_sixth_power_equals_one_l727_72746


namespace prove_x_equals_three_l727_72723

/-- Given real numbers a, b, c, d, and x, if (a - b) = (c + d) + 9, 
    (a + b) = (c - d) - x, and a - c = 3, then x = 3. -/
theorem prove_x_equals_three (a b c d x : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - x)
  (h3 : a - c = 3) : 
  x = 3 := by sorry

end prove_x_equals_three_l727_72723


namespace sum_proper_divisors_729_l727_72707

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x ≠ 0 ∧ n % x = 0)

theorem sum_proper_divisors_729 :
  (proper_divisors 729).sum id = 364 := by
  sorry

end sum_proper_divisors_729_l727_72707


namespace x_varies_as_z_to_four_thirds_l727_72704

/-- Given that x varies directly as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_z_to_four_thirds
  (h1 : ∃ (k : ℝ), ∀ (x y : ℝ), x = k * y^4)
  (h2 : ∃ (j : ℝ), ∀ (y z : ℝ), y = j * z^(1/3))
  : ∃ (m : ℝ), ∀ (x z : ℝ), x = m * z^(4/3) := by
  sorry

end x_varies_as_z_to_four_thirds_l727_72704


namespace vector_parallel_condition_l727_72777

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the parallel condition
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  are_parallel (a.1 + 2 * (b m).1, a.2 + 2 * (b m).2) (2 * a.1 - (b m).1, 2 * a.2 - (b m).2) →
  m = -1/2 :=
by sorry

end vector_parallel_condition_l727_72777


namespace probability_two_sunny_days_out_of_five_l727_72786

theorem probability_two_sunny_days_out_of_five :
  let n : ℕ := 5  -- total number of days
  let k : ℕ := 2  -- number of sunny days we want
  let p : ℚ := 1/4  -- probability of a sunny day (1 - probability of rain)
  let q : ℚ := 3/4  -- probability of a rainy day
  (n.choose k : ℚ) * p^k * q^(n - k) = 135/512 :=
sorry

end probability_two_sunny_days_out_of_five_l727_72786


namespace average_monthly_bill_l727_72710

/-- The average monthly bill for a family over 6 months, given the average for the first 4 months and the last 2 months. -/
theorem average_monthly_bill (avg_first_four : ℝ) (avg_last_two : ℝ) : 
  avg_first_four = 30 → avg_last_two = 24 → 
  (4 * avg_first_four + 2 * avg_last_two) / 6 = 28 := by
  sorry

end average_monthly_bill_l727_72710


namespace pen_cost_l727_72779

/-- The cost of a pen, given the cost of a pencil and the total cost of a specific number of pencils and pens. -/
theorem pen_cost (pencil_cost : ℝ) (total_cost : ℝ) (num_pencils : ℕ) (num_pens : ℕ) : 
  pencil_cost = 2.5 →
  total_cost = 291 →
  num_pencils = 38 →
  num_pens = 56 →
  (num_pencils : ℝ) * pencil_cost + (num_pens : ℝ) * ((total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens) = total_cost →
  (total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens = 3.5 := by
sorry

end pen_cost_l727_72779


namespace length_of_24_l727_72714

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 :
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end length_of_24_l727_72714


namespace subtraction_two_minus_three_l727_72751

theorem subtraction_two_minus_three : 2 - 3 = -1 := by
  sorry

end subtraction_two_minus_three_l727_72751


namespace log_27_3_l727_72766

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_27_3 : log 27 3 = 1/3 := by
  sorry

end log_27_3_l727_72766


namespace sum_of_x_solutions_l727_72759

theorem sum_of_x_solutions (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 289) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 289 ∧ x2^2 + y^2 = 289 ∧ x1 + x2 = 0 :=
sorry

end sum_of_x_solutions_l727_72759


namespace leap_stride_difference_l727_72794

/-- Represents the number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 44

/-- Represents the number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 12

/-- Represents the total number of poles -/
def total_poles : ℕ := 41

/-- Represents the total distance in feet between the first and last pole -/
def total_distance : ℕ := 5280

/-- Calculates the length of Elmer's stride in feet -/
def elmer_stride_length : ℚ :=
  total_distance / (elmer_strides * (total_poles - 1))

/-- Calculates the length of Oscar's leap in feet -/
def oscar_leap_length : ℚ :=
  total_distance / (oscar_leaps * (total_poles - 1))

/-- Theorem stating the difference between Oscar's leap and Elmer's stride -/
theorem leap_stride_difference : oscar_leap_length - elmer_stride_length = 8 := by
  sorry

end leap_stride_difference_l727_72794


namespace quadratic_equation_solutions_l727_72780

theorem quadratic_equation_solutions (x₁ x₂ : ℝ) :
  (x₁ = -1 ∧ x₂ = 3 ∧ x₁^2 - 2*x₁ - 3 = 0 ∧ x₂^2 - 2*x₂ - 3 = 0) →
  ∃ y₁ y₂ : ℝ, y₁ = 1 ∧ y₂ = -1 ∧ (2*y₁ + 1)^2 - 2*(2*y₁ + 1) - 3 = 0 ∧ (2*y₂ + 1)^2 - 2*(2*y₂ + 1) - 3 = 0 :=
by sorry

end quadratic_equation_solutions_l727_72780


namespace siblings_age_problem_l727_72770

theorem siblings_age_problem (b s : ℕ) : 
  (b - 3 = 7 * (s - 3)) →
  (b - 2 = 4 * (s - 2)) →
  (b - 1 = 3 * (s - 1)) →
  (b = (5 * s) / 2) →
  (b = 10 ∧ s = 4) :=
by sorry

end siblings_age_problem_l727_72770


namespace circus_ticket_problem_l727_72781

/-- Circus ticket problem -/
theorem circus_ticket_problem (num_kids : ℕ) (kid_ticket_price : ℚ) (total_cost : ℚ) :
  num_kids = 6 →
  kid_ticket_price = 5 →
  total_cost = 50 →
  ∃ (num_adults : ℕ),
    num_adults = 2 ∧
    total_cost = num_kids * kid_ticket_price + num_adults * (2 * kid_ticket_price) :=
by sorry

end circus_ticket_problem_l727_72781


namespace prove_some_number_l727_72708

theorem prove_some_number (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 35 * some_number * 35) :
  some_number = 21 := by
sorry

end prove_some_number_l727_72708


namespace right_triangle_segment_ratio_l727_72740

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_perp : r + s = c) 
  (h_r : r * c = a^2) 
  (h_s : s * c = b^2) 
  (h_ratio : a / b = 2 / 5) : 
  r / s = 4 / 25 := by 
sorry

end right_triangle_segment_ratio_l727_72740


namespace sum_of_slopes_on_parabola_l727_72797

/-- Given three points on a parabola with a focus satisfying certain conditions,
    prove that the sum of the slopes of the lines connecting these points is zero. -/
theorem sum_of_slopes_on_parabola (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  x₁^2 = 4*y₁ →
  x₂^2 = 4*y₂ →
  x₃^2 = 4*y₃ →
  x₁ + x₂ + x₃ = 0 →
  y₁ + y₂ + y₃ = 3 →
  (y₂ - y₁) / (x₂ - x₁) + (y₃ - y₁) / (x₃ - x₁) + (y₃ - y₂) / (x₃ - x₂) = 0 :=
by sorry

end sum_of_slopes_on_parabola_l727_72797


namespace rectangle_sides_l727_72747

theorem rectangle_sides (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) : 
  b = 3 * h → 
  h * b = 2 * (h + b) + Real.sqrt (h^2 + b^2) → 
  h = (8 + Real.sqrt 10) / 3 ∧ b = 8 + Real.sqrt 10 := by
  sorry

end rectangle_sides_l727_72747


namespace camping_payment_difference_l727_72754

/-- Represents the camping trip expenses and calculations --/
structure CampingExpenses where
  alan_paid : ℝ
  beth_paid : ℝ
  chris_paid : ℝ
  picnic_cost : ℝ
  total_cost : ℝ
  alan_share : ℝ
  beth_share : ℝ
  chris_share : ℝ

/-- Calculates the difference between what Alan and Beth need to pay Chris --/
def payment_difference (expenses : CampingExpenses) : ℝ :=
  (expenses.alan_share - expenses.alan_paid) - (expenses.beth_share - expenses.beth_paid)

/-- Theorem stating that the payment difference is 30 --/
theorem camping_payment_difference :
  ∃ (expenses : CampingExpenses),
    expenses.alan_paid = 110 ∧
    expenses.beth_paid = 140 ∧
    expenses.chris_paid = 190 ∧
    expenses.picnic_cost = 60 ∧
    expenses.total_cost = expenses.alan_paid + expenses.beth_paid + expenses.chris_paid + expenses.picnic_cost ∧
    expenses.alan_share = expenses.total_cost / 3 ∧
    expenses.beth_share = expenses.total_cost / 3 ∧
    expenses.chris_share = expenses.total_cost / 3 ∧
    payment_difference expenses = 30 := by
  sorry

end camping_payment_difference_l727_72754


namespace parents_years_in_america_before_aziz_birth_l727_72744

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_moved_to_america : ℕ := 1982
def parents_return_home : ℕ := 1995
def parents_return_america : ℕ := 1997

def aziz_birth_year : ℕ := current_year - aziz_age

def years_in_america : ℕ := aziz_birth_year - parents_moved_to_america

theorem parents_years_in_america_before_aziz_birth :
  years_in_america = 3 := by sorry

end parents_years_in_america_before_aziz_birth_l727_72744


namespace set_membership_implies_m_value_l727_72729

theorem set_membership_implies_m_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by sorry

end set_membership_implies_m_value_l727_72729


namespace y₁_less_than_y₂_l727_72769

/-- A linear function f(x) = -x + 5 -/
def f (x : ℝ) : ℝ := -x + 5

/-- P₁ is a point on the graph of f with x-coordinate -2 -/
def P₁ (y₁ : ℝ) : Prop := f (-2) = y₁

/-- P₂ is a point on the graph of f with x-coordinate -3 -/
def P₂ (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem: If P₁(-2, y₁) and P₂(-3, y₂) are points on the graph of f, then y₁ < y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁ y₁) (h₂ : P₂ y₂) : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l727_72769


namespace intersection_M_N_l727_72758

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end intersection_M_N_l727_72758


namespace initial_milk_cost_initial_milk_cost_is_four_l727_72756

/-- Calculates the initial cost of milk given the grocery shopping scenario --/
theorem initial_milk_cost (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) 
  (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) 
  (milk_discount_factor : ℝ) (money_left : ℝ) : ℝ :=
  let total_spent := total_money - money_left
  let banana_cost := banana_cost_per_pound * banana_pounds
  let discounted_detergent_cost := detergent_cost - detergent_coupon
  let non_milk_cost := bread_cost + banana_cost + discounted_detergent_cost
  let milk_cost := total_spent - non_milk_cost
  milk_cost / milk_discount_factor

/-- The initial cost of milk is $4 --/
theorem initial_milk_cost_is_four :
  initial_milk_cost 20 3.5 10.25 0.75 2 1.25 0.5 4 = 4 := by
  sorry

end initial_milk_cost_initial_milk_cost_is_four_l727_72756


namespace correct_operation_result_l727_72773

theorem correct_operation_result (x : ℝ) : (x - 9) / 3 = 43 → (x - 3) / 9 = 15 := by
  sorry

end correct_operation_result_l727_72773


namespace billy_weight_l727_72787

theorem billy_weight (carl_weight brad_weight billy_weight : ℕ) 
  (h1 : brad_weight = carl_weight + 5)
  (h2 : billy_weight = brad_weight + 9)
  (h3 : carl_weight = 145) :
  billy_weight = 159 := by
  sorry

end billy_weight_l727_72787


namespace no_x_squared_term_l727_72730

/-- 
Given the algebraic expression (x-2)(ax²-x+1), this theorem states that
the coefficient of x² in the expanded form is zero if and only if a = -1/2.
-/
theorem no_x_squared_term (x a : ℝ) : 
  (x - 2) * (a * x^2 - x + 1) = a * x^3 + 3 * x - 2 ↔ a = -1/2 := by
sorry

end no_x_squared_term_l727_72730


namespace equidistant_point_x_coordinate_l727_72752

/-- The x-coordinate of the point on the x-axis equidistant from C(-3, 0) and D(2, 5) is 2 -/
theorem equidistant_point_x_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (2, 5)
  ∃ x : ℝ, x = 2 ∧
    (x - C.1)^2 + C.2^2 = (x - D.1)^2 + D.2^2 :=
by sorry

end equidistant_point_x_coordinate_l727_72752


namespace denny_initial_followers_l727_72741

/-- Calculates the initial number of followers given the daily increase, total unfollows, 
    final follower count, and number of days in a year. -/
def initial_followers (daily_increase : ℕ) (total_unfollows : ℕ) (final_count : ℕ) (days_in_year : ℕ) : ℕ :=
  final_count - (daily_increase * days_in_year) + total_unfollows

/-- Proves that given the specified conditions, the initial number of followers is 100000. -/
theorem denny_initial_followers : 
  initial_followers 1000 20000 445000 365 = 100000 := by
  sorry

end denny_initial_followers_l727_72741


namespace exactly_three_blue_marbles_l727_72791

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exactly_three_blue_marbles :
  Nat.choose num_picks num_blue_picks *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_picks - num_blue_picks)) =
  640 / 1547 := by sorry

end exactly_three_blue_marbles_l727_72791


namespace fishing_result_l727_72701

/-- The number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_per_catfish : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_per_catfish
  let henry_keeps := henry_total / 2
  will_total + henry_keeps

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

end fishing_result_l727_72701


namespace inverse_at_five_l727_72775

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State that f has an inverse
def f_inv : ℝ → ℝ := sorry

-- Assume f_inv is the inverse of f
axiom f_inverse (x : ℝ) : f (f_inv x) = x
axiom inv_f (x : ℝ) : f_inv (f x) = x

-- Theorem to prove
theorem inverse_at_five : f_inv 5 = 3 := by
  sorry

end inverse_at_five_l727_72775


namespace probability_calculation_l727_72796

/-- The number of people with blocks -/
def num_people : ℕ := 3

/-- The number of blocks each person has -/
def blocks_per_person : ℕ := 6

/-- The number of empty boxes -/
def num_boxes : ℕ := 5

/-- The maximum number of blocks a person can place in a box -/
def max_blocks_per_person_per_box : ℕ := 2

/-- The maximum total number of blocks allowed in a box -/
def max_blocks_per_box : ℕ := 4

/-- The number of ways each person can distribute their blocks -/
def distribution_ways : ℕ := (num_boxes + blocks_per_person - 1).choose (blocks_per_person - 1)

/-- The number of favorable distributions for a specific box getting all blocks of the same color -/
def favorable_distributions : ℕ := blocks_per_person

/-- The probability that at least one box has all blocks of the same color -/
def probability_same_color : ℝ := 1.86891e-6

theorem probability_calculation :
  1 - num_boxes * (favorable_distributions : ℝ) / (distribution_ways ^ num_people : ℝ) = probability_same_color := by
  sorry

end probability_calculation_l727_72796


namespace adjacent_supplementary_angles_l727_72709

theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  (angle1 + angle2 = 180) → (angle1 = 80) → (angle2 = 100) := by
  sorry

end adjacent_supplementary_angles_l727_72709


namespace amy_flash_drive_files_l727_72778

/-- The number of files on Amy's flash drive -/
def total_files (music_files video_files picture_files : Float) : Float :=
  music_files + video_files + picture_files

/-- Theorem stating the total number of files on Amy's flash drive -/
theorem amy_flash_drive_files : 
  total_files 4.0 21.0 23.0 = 48.0 := by
  sorry

end amy_flash_drive_files_l727_72778


namespace greatest_x_quadratic_inequality_l727_72760

theorem greatest_x_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 4 ∧ 
  (∀ (x : ℝ), -2 * x^2 + 12 * x - 16 ≥ 0 → x ≤ x_max) ∧
  (-2 * x_max^2 + 12 * x_max - 16 ≥ 0) :=
by sorry

end greatest_x_quadratic_inequality_l727_72760


namespace positive_sequence_existence_l727_72765

theorem positive_sequence_existence :
  ∃ (a : ℕ → ℝ) (a₁ : ℝ), 
    (∀ n, a n > 0) ∧
    (∀ n, a (n + 2) = a n - a (n + 1)) ∧
    (a₁ > 0) ∧
    (∀ n, a n = a₁ * ((Real.sqrt 5 - 1) / 2) ^ (n - 1)) :=
by sorry

end positive_sequence_existence_l727_72765


namespace serenas_age_problem_l727_72782

/-- Proves that in 6 years, Serena's mother will be three times as old as Serena. -/
theorem serenas_age_problem (serena_age : ℕ) (mother_age : ℕ) 
  (h1 : serena_age = 9) (h2 : mother_age = 39) : 
  ∃ (years : ℕ), years = 6 ∧ mother_age + years = 3 * (serena_age + years) := by
  sorry

end serenas_age_problem_l727_72782


namespace field_trip_problem_l727_72762

/-- Given a field trip with vans and buses, calculates the number of people in each van. -/
def peoplePerVan (numVans : ℕ) (numBuses : ℕ) (peoplePerBus : ℕ) (totalPeople : ℕ) : ℕ :=
  (totalPeople - numBuses * peoplePerBus) / numVans

theorem field_trip_problem :
  peoplePerVan 6 8 18 180 = 6 := by
  sorry

end field_trip_problem_l727_72762


namespace inequality_proof_l727_72731

theorem inequality_proof (x : ℝ) (h1 : 0 < x) (h2 : x < 20) :
  Real.sqrt x + Real.sqrt (20 - x) ≤ 2 * Real.sqrt 10 := by
  sorry

end inequality_proof_l727_72731


namespace b_share_is_108_l727_72717

/-- Represents the share ratio of partners A, B, and C -/
structure ShareRatio where
  a : Rat
  b : Rat
  c : Rat

/-- Represents the capital contribution of partners over time -/
structure CapitalContribution where
  a : Rat
  b : Rat
  c : Rat

def initial_ratio : ShareRatio :=
  { a := 1/2, b := 1/3, c := 1/4 }

def total_profit : ℚ := 378

def months_before_withdrawal : ℕ := 2
def total_months : ℕ := 12

def capital_contribution (r : ShareRatio) : CapitalContribution :=
  { a := r.a * months_before_withdrawal + (r.a / 2) * (total_months - months_before_withdrawal),
    b := r.b * total_months,
    c := r.c * total_months }

theorem b_share_is_108 (r : ShareRatio) (cc : CapitalContribution) :
  r = initial_ratio →
  cc = capital_contribution r →
  (cc.b / (cc.a + cc.b + cc.c)) * total_profit = 108 :=
by sorry

end b_share_is_108_l727_72717


namespace inverse_exists_iff_a_eq_zero_l727_72789

-- Define the function f(x) = (x - a)|x|
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * abs x

-- State the theorem
theorem inverse_exists_iff_a_eq_zero (a : ℝ) :
  Function.Injective (f a) ↔ a = 0 := by sorry

end inverse_exists_iff_a_eq_zero_l727_72789


namespace shell_division_impossibility_l727_72735

theorem shell_division_impossibility : ¬ ∃ (n : ℕ), 
  (637 - n) % 3 = 0 ∧ (n + 1 : ℕ) = (637 - n) / 3 := by
  sorry

end shell_division_impossibility_l727_72735
