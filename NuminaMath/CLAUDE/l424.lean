import Mathlib

namespace install_time_proof_l424_42400

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install the remaining windows is 48 hours -/
theorem install_time_proof (total : ℕ) (installed : ℕ) (time_per_window : ℕ)
  (h1 : total = 14)
  (h2 : installed = 8)
  (h3 : time_per_window = 8) :
  time_to_install_remaining total installed time_per_window = 48 := by
  sorry

#eval time_to_install_remaining 14 8 8

end install_time_proof_l424_42400


namespace equal_roots_quadratic_l424_42455

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 10 = 0 → y = x) ↔ 
  (k = 2 - 2 * Real.sqrt 30 ∨ k = -2 - 2 * Real.sqrt 30) :=
by sorry

end equal_roots_quadratic_l424_42455


namespace circles_intersect_l424_42497

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Theorem stating that the circles intersect
theorem circles_intersect : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end circles_intersect_l424_42497


namespace hidden_dots_count_l424_42428

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := (List.range 6).sum + 6

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 2, 3, 4, 5, 4, 6, 5, 3]

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The number of visible faces -/
def numberOfVisibleFaces : ℕ := 9

theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - visibleFaces.sum = 51 := by
  sorry

end hidden_dots_count_l424_42428


namespace belle_treats_cost_l424_42486

/-- The cost of feeding Belle her treats for a week -/
def cost_per_week : ℚ :=
  let biscuits_per_day : ℕ := 4
  let bones_per_day : ℕ := 2
  let biscuit_cost : ℚ := 1/4
  let bone_cost : ℚ := 1
  let days_per_week : ℕ := 7
  (biscuits_per_day * biscuit_cost + bones_per_day * bone_cost) * days_per_week

/-- Theorem stating that the cost of feeding Belle her treats for a week is $21 -/
theorem belle_treats_cost : cost_per_week = 21 := by
  sorry

end belle_treats_cost_l424_42486


namespace correct_statements_count_l424_42423

theorem correct_statements_count (x : ℝ) : 
  (((x > 0) → (x^2 > 0)) ∧ ((x^2 ≤ 0) → (x ≤ 0)) ∧ ¬((x ≤ 0) → (x^2 ≤ 0))) :=
by sorry

end correct_statements_count_l424_42423


namespace min_value_of_trig_function_l424_42466

open Real

theorem min_value_of_trig_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (∀ y, 0 < y ∧ y < π / 2 → 
    (1 + cos (2 * y) + 8 * sin y ^ 2) / sin (2 * y) ≥ 
    (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x)) →
  (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x) = 4 :=
by sorry

end min_value_of_trig_function_l424_42466


namespace M_subset_P_l424_42470

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def P : Set ℝ := Set.univ

-- State the theorem
theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l424_42470


namespace at_least_one_negative_l424_42415

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l424_42415


namespace square_circle_union_area_l424_42453

/-- The area of the union of a square with side length 12 and a circle with radius 6 
    centered at the center of the square is equal to 144. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 6
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  square_area = circle_area + 144 := by
  sorry

end square_circle_union_area_l424_42453


namespace lilys_books_l424_42447

theorem lilys_books (books_last_month : ℕ) : 
  books_last_month + 2 * books_last_month = 12 → books_last_month = 4 := by
  sorry

end lilys_books_l424_42447


namespace partner_C_investment_l424_42498

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  investment_A : ℕ
  investment_B : ℕ
  profit_share_B : ℕ
  profit_diff_AC : ℕ

/-- Calculates the investment of partner C given the business partnership details --/
def calculate_investment_C (bp : BusinessPartnership) : ℕ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that given the specific business partnership details, 
    partner C's investment is 120000 --/
theorem partner_C_investment 
  (bp : BusinessPartnership) 
  (h1 : bp.investment_A = 8000)
  (h2 : bp.investment_B = 10000)
  (h3 : bp.profit_share_B = 1400)
  (h4 : bp.profit_diff_AC = 560) : 
  calculate_investment_C bp = 120000 := by
  sorry

end partner_C_investment_l424_42498


namespace union_of_sets_l424_42469

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | x^2 - 4*x < 0}

theorem union_of_sets : set_A ∪ set_B = Set.Ioo (-2) 4 := by sorry

end union_of_sets_l424_42469


namespace inequality_proof_l424_42473

theorem inequality_proof (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > a*b ∧ a*b > a :=
by sorry

end inequality_proof_l424_42473


namespace intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l424_42427

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y - a + 2 = 0
def l₂ (a x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Theorem for the intersection point when a = 1
theorem intersection_point_when_a_is_one :
  ∃ (x y : ℝ), l₁ 1 x y ∧ l₂ 1 x y ∧ x = -4 ∧ y = 3 :=
sorry

-- Definition of parallel lines
def parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (1 : ℝ) / (a : ℝ) = k * (2 * a) / (a + 3) ∧
  (a ≠ -3)

-- Theorem for parallel lines when a = 3/2
theorem parallel_when_a_is_three_halves :
  parallel (3/2) :=
sorry

end intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l424_42427


namespace triangles_with_fixed_vertex_l424_42426

theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 9) :
  Nat.choose (n - 1) 2 = 28 :=
sorry

end triangles_with_fixed_vertex_l424_42426


namespace triangle_theorem_l424_42459

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  (t.A = 2 * t.B → t.C = 5 * Real.pi / 8) ∧ 
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry


end triangle_theorem_l424_42459


namespace complex_sum_equals_two_l424_42440

theorem complex_sum_equals_two (z : ℂ) (h : z^7 = 1) (h2 : z = Complex.exp (2 * Real.pi * Complex.I / 7)) : 
  (z^2 / (1 + z^3)) + (z^4 / (1 + z^6)) + (z^6 / (1 + z^9)) = 2 := by
  sorry

end complex_sum_equals_two_l424_42440


namespace chicken_multiple_l424_42452

theorem chicken_multiple (total chickens : ℕ) (colten_chickens : ℕ) (m : ℕ) : 
  total = 383 →
  colten_chickens = 37 →
  (∃ (quentin skylar : ℕ), 
    quentin + skylar + colten_chickens = total ∧
    quentin = 2 * skylar + 25 ∧
    skylar = m * colten_chickens - 4) →
  m = 3 := by
  sorry

end chicken_multiple_l424_42452


namespace triangle_properties_l424_42462

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 1 ∧ Real.cos t.C + (2 * t.a + t.c) * Real.cos t.B = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = 1/2 * t.a * t.c * Real.sin t.B → s ≤ Real.sqrt 3 / 12) :=
sorry

end triangle_properties_l424_42462


namespace infinitely_many_all_off_infinitely_many_never_all_off_l424_42414

-- Define the lamp state as a list of booleans
def LampState := List Bool

-- Define the state modification function
def modifyState (state : LampState) : LampState :=
  sorry

-- Define the initial state
def initialState (n : Nat) : LampState :=
  sorry

-- Define a predicate to check if all lamps are off
def allLampsOff (state : LampState) : Prop :=
  sorry

-- Define a function to evolve the state
def evolveState (n : Nat) (steps : Nat) : LampState :=
  sorry

theorem infinitely_many_all_off :
  ∃ S : Set Nat, (∀ n ∈ S, n ≥ 2) ∧ Set.Infinite S ∧
  ∀ n ∈ S, ∃ k : Nat, allLampsOff (evolveState n k) :=
sorry

theorem infinitely_many_never_all_off :
  ∃ T : Set Nat, (∀ n ∈ T, n ≥ 2) ∧ Set.Infinite T ∧
  ∀ n ∈ T, ∀ k : Nat, ¬(allLampsOff (evolveState n k)) :=
sorry

end infinitely_many_all_off_infinitely_many_never_all_off_l424_42414


namespace equation_solutions_l424_42408

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 6 * x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x + 2)^2 = 3 * (x + 2)
  let sol1 : Set ℝ := {(-3 + Real.sqrt 3) / 2, (-3 - Real.sqrt 3) / 2}
  let sol2 : Set ℝ := {-2, 1}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end equation_solutions_l424_42408


namespace negation_of_proposition_l424_42448

theorem negation_of_proposition :
  (¬(∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0)) ↔
  (∀ a b : ℝ, a^2 + b^2 = 0 → a ≠ 0 ∨ b ≠ 0) :=
by sorry

end negation_of_proposition_l424_42448


namespace school_enrollment_increase_l424_42407

-- Define the variables and constants
def last_year_total : ℕ := 4000
def last_year_YY : ℕ := 2400
def XX_percent_increase : ℚ := 7 / 100
def extra_growth_XX : ℕ := 40

-- Define the theorem
theorem school_enrollment_increase : 
  ∃ (p : ℚ), 
    (p ≥ 0) ∧ 
    (p ≤ 1) ∧
    (XX_percent_increase * (last_year_total - last_year_YY) = 
     (p * last_year_YY) + extra_growth_XX) ∧
    (p = 3 / 100) := by
  sorry

end school_enrollment_increase_l424_42407


namespace book_sale_discount_l424_42456

/-- Calculates the discount percentage for a book sale --/
theorem book_sale_discount (cost : ℝ) (markup_percent : ℝ) (profit_percent : ℝ) 
  (h_cost : cost = 50)
  (h_markup : markup_percent = 30)
  (h_profit : profit_percent = 17) : 
  let marked_price := cost * (1 + markup_percent / 100)
  let selling_price := cost * (1 + profit_percent / 100)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 10 := by
  sorry

end book_sale_discount_l424_42456


namespace notebook_cost_l424_42494

theorem notebook_cost (book_cost : ℝ) (binders_cost : ℝ) (num_notebooks : ℕ) (total_cost : ℝ)
  (h1 : book_cost = 16)
  (h2 : binders_cost = 6)
  (h3 : num_notebooks = 6)
  (h4 : total_cost = 28)
  : (total_cost - (book_cost + binders_cost)) / num_notebooks = 1 := by
  sorry

end notebook_cost_l424_42494


namespace right_triangle_hypotenuse_l424_42403

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 34 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 62/4 := by
sorry

end right_triangle_hypotenuse_l424_42403


namespace vasyas_birthday_l424_42461

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : next_day (next_day today) = DayOfWeek.Sunday) -- Sunday is the day after tomorrow
  (h2 : ∃ birthday : DayOfWeek, next_day birthday = today) -- Today is the day after Vasya's birthday
  : ∃ birthday : DayOfWeek, birthday = DayOfWeek.Thursday := by
  sorry

end vasyas_birthday_l424_42461


namespace physical_exercise_test_results_l424_42443

/-- Represents a school in the physical exercise test --/
structure School where
  name : String
  total_students : Nat
  sampled_students : Nat
  average_score : Float
  median_score : Float
  mode_score : Nat

/-- Represents the score distribution for a school --/
structure ScoreDistribution where
  school : School
  scores : List (Nat × Nat)  -- (score_range_start, count)

theorem physical_exercise_test_results 
  (school_a school_b : School)
  (dist_a : ScoreDistribution)
  (h1 : school_a.name = "School A")
  (h2 : school_b.name = "School B")
  (h3 : school_a.total_students = 180)
  (h4 : school_b.total_students = 180)
  (h5 : school_a.sampled_students = 30)
  (h6 : school_b.sampled_students = 30)
  (h7 : school_a.average_score = 96.35)
  (h8 : school_a.mode_score = 99)
  (h9 : school_b.average_score = 95.85)
  (h10 : school_b.median_score = 97.5)
  (h11 : school_b.mode_score = 99)
  (h12 : dist_a.school = school_a)
  (h13 : dist_a.scores = [(90, 2), (92, 3), (94, 5), (96, 10), (98, 10)]) :
  school_a.median_score = 96.5 ∧ 
  (((school_a.total_students * 20) / 30 : Nat) * 2 - 100 = 140) := by
  sorry

end physical_exercise_test_results_l424_42443


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l424_42418

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (5 + Real.sqrt 1) / 2
  let r₂ := (5 - Real.sqrt 1) / 2
  r₁ + r₂ = 5 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l424_42418


namespace square_EFGH_area_l424_42490

theorem square_EFGH_area : 
  ∀ (original_side_length : ℝ) (EFGH_side_length : ℝ),
  original_side_length = 8 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length^2 = 256 :=
by sorry

end square_EFGH_area_l424_42490


namespace factorization_proof_l424_42444

theorem factorization_proof (x : ℝ) : -2 * x^2 + 18 = -2 * (x + 3) * (x - 3) := by
  sorry

end factorization_proof_l424_42444


namespace f_monotone_intervals_f_B_range_l424_42454

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotone_intervals (k : ℤ) :
  is_monotone_increasing f (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) := by sorry

theorem f_B_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  b * Real.cos (2 * A) = b * Real.cos A - a * Real.sin B →
  ∃ x, f B = x ∧ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2 := by sorry

end f_monotone_intervals_f_B_range_l424_42454


namespace remainder_theorem_l424_42449

theorem remainder_theorem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 39) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end remainder_theorem_l424_42449


namespace quadratic_factorization_l424_42465

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) → 
  c + 2 * d = -2 := by
sorry

end quadratic_factorization_l424_42465


namespace power_equation_solution_l424_42487

theorem power_equation_solution : ∃ x : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * (15 ^ x) ∧ x = 5 := by
  sorry

end power_equation_solution_l424_42487


namespace age_problem_l424_42438

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 42 →
  b = 16 := by
sorry

end age_problem_l424_42438


namespace hundredths_place_of_five_eighths_l424_42405

theorem hundredths_place_of_five_eighths : ∃ (n : ℕ), (5 : ℚ) / 8 = (n * 100 + 20 : ℚ) / 1000 := by
  sorry

end hundredths_place_of_five_eighths_l424_42405


namespace shorter_tree_height_l424_42478

theorem shorter_tree_height (h1 h2 : ℝ) : 
  h2 = h1 + 20 →  -- One tree is 20 feet taller
  h1 / h2 = 5 / 7 →  -- Heights are in ratio 5:7
  h1 + h2 = 240 →  -- Sum of heights is 240 feet
  h1 = 110 :=  -- Shorter tree is 110 feet tall
by sorry

end shorter_tree_height_l424_42478


namespace beths_shopping_multiple_l424_42420

/-- The problem of Beth's shopping for peas and corn -/
theorem beths_shopping_multiple (peas corn : ℕ) (multiple : ℚ) 
  (h1 : peas = corn * multiple + 15)
  (h2 : peas = 35)
  (h3 : corn = 10) :
  multiple = 2 := by
  sorry

end beths_shopping_multiple_l424_42420


namespace largest_difference_l424_42493

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 2005^2006)
  (hB : B = 2005^2006)
  (hC : C = 2004 * 2005^2005)
  (hD : D = 3 * 2005^2005)
  (hE : E = 2005^2005)
  (hF : F = 2005^2004) :
  A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by sorry

end largest_difference_l424_42493


namespace tricubic_properties_l424_42404

def tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

def exactly_one_tricubic (n : ℕ) : Prop :=
  (tricubic n ∧ ¬tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (¬tricubic n ∧ tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (¬tricubic n ∧ ¬tricubic (n+2) ∧ tricubic (n+28))

def exactly_two_tricubic (n : ℕ) : Prop :=
  (tricubic n ∧ tricubic (n+2) ∧ ¬tricubic (n+28)) ∨
  (tricubic n ∧ ¬tricubic (n+2) ∧ tricubic (n+28)) ∨
  (¬tricubic n ∧ tricubic (n+2) ∧ tricubic (n+28))

def all_three_tricubic (n : ℕ) : Prop :=
  tricubic n ∧ tricubic (n+2) ∧ tricubic (n+28)

theorem tricubic_properties :
  (∃ f : ℕ → ℕ, ∀ k, k < f k ∧ exactly_one_tricubic (f k)) ∧
  (∃ g : ℕ → ℕ, ∀ k, k < g k ∧ exactly_two_tricubic (g k)) ∧
  (∃ h : ℕ → ℕ, ∀ k, k < h k ∧ all_three_tricubic (h k)) := by
  sorry

end tricubic_properties_l424_42404


namespace distance_between_trees_l424_42402

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : 
  yard_length = 273 ∧ num_trees = 14 → 
  (yard_length : ℚ) / (num_trees - 1 : ℚ) = 21 := by
  sorry

end distance_between_trees_l424_42402


namespace project_hours_difference_l424_42483

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 117 →
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 65 := by
  sorry

end project_hours_difference_l424_42483


namespace hannahs_pay_l424_42457

/-- Calculates the final pay for an employee given their hourly rate, hours worked, late penalty, and number of times late. -/
def calculate_final_pay (hourly_rate : ℕ) (hours_worked : ℕ) (late_penalty : ℕ) (times_late : ℕ) : ℕ :=
  hourly_rate * hours_worked - late_penalty * times_late

/-- Proves that Hannah's final pay is $525 given her work conditions. -/
theorem hannahs_pay :
  calculate_final_pay 30 18 5 3 = 525 := by
  sorry

end hannahs_pay_l424_42457


namespace midpoint_coordinate_product_l424_42410

/-- Given a line segment CD with midpoint N and endpoint C, proves that the product of D's coordinates is 39 -/
theorem midpoint_coordinate_product (C N D : ℝ × ℝ) : 
  C = (5, 3) → N = (4, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = 39 := by
  sorry

#check midpoint_coordinate_product

end midpoint_coordinate_product_l424_42410


namespace remainder_67_power_67_plus_67_mod_68_l424_42484

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_67_power_67_plus_67_mod_68_l424_42484


namespace f_odd_and_increasing_l424_42446

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end f_odd_and_increasing_l424_42446


namespace equation_solution_l424_42463

theorem equation_solution (x : ℝ) : 
  (3 / (x^2 + x) - x^2 = 2 + x) → (2*x^2 + 2*x = 2) :=
by
  sorry

end equation_solution_l424_42463


namespace collinear_dots_probability_l424_42433

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots from horizontal or vertical lines -/
def horizontalVerticalSets : ℕ := 2 * gridSize

/-- The number of ways to choose 4 collinear dots from diagonal lines -/
def diagonalSets : ℕ := 2 * (Nat.choose gridSize chosenDots)

/-- The total number of ways to choose 4 collinear dots -/
def totalCollinearSets : ℕ := horizontalVerticalSets + diagonalSets

/-- Theorem: The probability of selecting 4 collinear dots from a 5x5 grid 
    when choosing 4 dots at random is 4/2530 -/
theorem collinear_dots_probability : 
  (totalCollinearSets : ℚ) / (Nat.choose totalDots chosenDots) = 4 / 2530 := by
  sorry

end collinear_dots_probability_l424_42433


namespace triangle_ratio_equals_two_l424_42437

noncomputable def triangle_ratio (A B C : ℝ) (a b c : ℝ) : ℝ :=
  (a + b - c) / (Real.sin A + Real.sin B - Real.sin C)

theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = π / 3)  -- 60° in radians
  (h2 : a = Real.sqrt 3) :
  triangle_ratio A B C a b c = 2 := by
  sorry

end triangle_ratio_equals_two_l424_42437


namespace cos_2alpha_special_value_l424_42479

theorem cos_2alpha_special_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (α - π/4) = 1/3) : 
  Real.cos (2*α) = -4*Real.sqrt 2/9 := by
sorry

end cos_2alpha_special_value_l424_42479


namespace quadratic_equation_condition_l424_42496

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 1| = 2 ∧ m + 1 ≠ 0) ↔ m = 3 := by sorry

end quadratic_equation_condition_l424_42496


namespace min_squares_for_25x25_grid_l424_42401

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ
  total_squares : ℕ

/-- Calculates the minimum number of 1x1 squares needed to create an image of a square grid -/
def min_squares_for_image (grid : SquareGrid) : ℕ :=
  let perimeter := 4 * grid.size - 4
  let interior := (grid.size - 2) * (grid.size - 2)
  let dominos := interior / 2
  perimeter + dominos

/-- Theorem stating the minimum number of squares needed for a 25x25 grid -/
theorem min_squares_for_25x25_grid :
  ∃ (grid : SquareGrid), grid.size = 25 ∧ grid.total_squares = 625 ∧ min_squares_for_image grid = 360 := by
  sorry

end min_squares_for_25x25_grid_l424_42401


namespace complex_fraction_simplification_l424_42409

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i := by
  sorry

end complex_fraction_simplification_l424_42409


namespace isabella_read_250_pages_l424_42491

/-- The number of pages Isabella read in a week -/
def total_pages (pages_first_three : ℕ) (pages_next_three : ℕ) (pages_last_day : ℕ) : ℕ :=
  3 * pages_first_three + 3 * pages_next_three + pages_last_day

/-- Theorem stating that Isabella read 250 pages in total -/
theorem isabella_read_250_pages : 
  total_pages 36 44 10 = 250 := by
  sorry

#check isabella_read_250_pages

end isabella_read_250_pages_l424_42491


namespace soccer_balls_per_class_l424_42430

theorem soccer_balls_per_class 
  (num_schools : ℕ)
  (elementary_classes_per_school : ℕ)
  (middle_classes_per_school : ℕ)
  (total_soccer_balls : ℕ)
  (h1 : num_schools = 2)
  (h2 : elementary_classes_per_school = 4)
  (h3 : middle_classes_per_school = 5)
  (h4 : total_soccer_balls = 90) :
  total_soccer_balls / (num_schools * (elementary_classes_per_school + middle_classes_per_school)) = 5 :=
by sorry

end soccer_balls_per_class_l424_42430


namespace geometric_sum_first_8_terms_l424_42488

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 9840/6561 := by
sorry

end geometric_sum_first_8_terms_l424_42488


namespace square_area_from_diagonal_l424_42450

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 * Real.sqrt 2 → area = 144 → 
  diagonal^2 / 2 = area := by sorry

end square_area_from_diagonal_l424_42450


namespace unique_five_digit_number_l424_42431

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def move_first_digit_to_end (n : ℕ) : ℕ :=
  (n % 10000) * 10 + (n / 10000)

theorem unique_five_digit_number : ∃! n : ℕ,
  is_five_digit n ∧
  move_first_digit_to_end n = n + 34767 ∧
  move_first_digit_to_end n + n = 86937 ∧
  n = 26035 := by
sorry

end unique_five_digit_number_l424_42431


namespace unpainted_cubes_4x4x4_l424_42474

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted strip on a cube face --/
structure PaintedStrip where
  width : ℕ
  height : ℕ

/-- Calculates the number of unpainted unit cubes in a cube with painted strips --/
def unpainted_cubes (c : Cube 4) (strip : PaintedStrip) : ℕ :=
  sorry

theorem unpainted_cubes_4x4x4 :
  ∀ (c : Cube 4) (strip : PaintedStrip),
    strip.width = 2 ∧ strip.height = c.side_length →
    unpainted_cubes c strip = 40 := by
  sorry

end unpainted_cubes_4x4x4_l424_42474


namespace ant_journey_l424_42434

-- Define the plane and points A and B
variable (Plane : Type) (A B : Plane)

-- Define the distance functions from A and B
variable (distA distB : ℝ → ℝ)

-- Define the conditions
variable (h1 : distA 7 = 5)
variable (h2 : distB 7 = 3)
variable (h3 : distB 0 = 0)
variable (h4 : distA 0 = 4)

-- Define the distance between A and B
def dist_AB : ℝ := 4

-- Define the theorem
theorem ant_journey :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ distA t1 = distB t1 ∧ distA t2 = distB t2) ∧
  (dist_AB = 4) ∧
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ 0 ≤ t3 ∧ t3 ≤ 9 ∧
    distA t1 + distB t1 = dist_AB ∧
    distA t2 + distB t2 = dist_AB ∧
    distA t3 + distB t3 = dist_AB) ∧
  (∃ d : ℝ, d = 8 ∧ 
    d = |distA 3 - distA 0| + |distA 5 - distA 3| + |distA 7 - distA 5| + |distA 9 - distA 7|) :=
by sorry

end ant_journey_l424_42434


namespace intersection_locus_l424_42422

/-- The locus of the intersection point of two lines in a Cartesian coordinate system -/
theorem intersection_locus (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  ∀ (x y : ℝ), 
  (∃ c : ℝ, c ≠ 0 ∧ 
    (y = (a / c) * x) ∧ 
    (x / b + y / c = 1)) →
  ((x - b / 2)^2 / (b^2 / 4) + y^2 / (a * b / 4) = 1) :=
by sorry

end intersection_locus_l424_42422


namespace clever_calculation_l424_42406

theorem clever_calculation :
  (1978 + 250 + 1022 + 750 = 4000) ∧
  (454 + 999 * 999 + 545 = 999000) ∧
  (999 + 998 + 997 + 996 + 1004 + 1003 + 1002 + 1001 = 8000) := by
  sorry

end clever_calculation_l424_42406


namespace cubic_fraction_sum_l424_42439

theorem cubic_fraction_sum (a b : ℝ) (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 := by
  sorry

end cubic_fraction_sum_l424_42439


namespace equality_check_l424_42482

theorem equality_check : 
  ((-2 : ℤ)^3 ≠ -2 * 3) ∧ 
  (2^3 ≠ 3^2) ∧ 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) := by
  sorry

end equality_check_l424_42482


namespace calculation_proof_inequality_system_solution_l424_42411

-- Part 1
theorem calculation_proof :
  |Real.sqrt 5 - 3| + (1/2)⁻¹ - Real.sqrt 20 + Real.sqrt 3 * Real.cos (30 * π / 180) = 13/2 - 3 * Real.sqrt 5 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x ∧ (1 + 2*x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end calculation_proof_inequality_system_solution_l424_42411


namespace factor_implies_b_value_l424_42432

/-- The polynomial Q(x) -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 5

/-- Theorem: If x - 5 is a factor of Q(x), then b = -41 -/
theorem factor_implies_b_value (b : ℝ) : 
  (∀ x, Q b x = 0 ↔ x = 5) → b = -41 := by
  sorry

end factor_implies_b_value_l424_42432


namespace regular_pay_limit_l424_42442

/-- The problem of finding the limit for regular pay. -/
theorem regular_pay_limit (regular_rate : ℝ) (overtime_rate : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  overtime_rate = 2 * regular_rate →
  total_pay = 186 →
  overtime_hours = 11 →
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * overtime_rate = total_pay ∧
    regular_hours = 40 :=
by sorry

end regular_pay_limit_l424_42442


namespace prob_ace_ten_queen_correct_l424_42471

/-- The probability of drawing an Ace, then a 10, then a Queen from a standard 52-card deck without replacement -/
def prob_ace_ten_queen : ℚ := 8 / 16575

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of 10s in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

theorem prob_ace_ten_queen_correct (d : Deck) : 
  (num_aces : ℚ) / 52 * (num_tens : ℚ) / 51 * (num_queens : ℚ) / 50 = prob_ace_ten_queen :=
sorry

end prob_ace_ten_queen_correct_l424_42471


namespace x_value_l424_42429

theorem x_value : ∃ x : ℝ, (3 * x = (20 - x) + 20) ∧ (x = 10) := by
  sorry

end x_value_l424_42429


namespace intersection_of_P_and_Q_l424_42458

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end intersection_of_P_and_Q_l424_42458


namespace rank_from_bottom_calculation_l424_42480

/-- Represents a student's ranking in a class. -/
structure StudentRanking where
  totalStudents : Nat
  rankFromTop : Nat
  rankFromBottom : Nat

/-- Calculates the rank from the bottom given the total number of students and rank from the top. -/
def calculateRankFromBottom (total : Nat) (rankFromTop : Nat) : Nat :=
  total - rankFromTop + 1

/-- Theorem stating that for a class of 53 students, a student ranking 5th from the top
    will rank 49th from the bottom. -/
theorem rank_from_bottom_calculation (s : StudentRanking)
    (h1 : s.totalStudents = 53)
    (h2 : s.rankFromTop = 5)
    (h3 : s.rankFromBottom = calculateRankFromBottom s.totalStudents s.rankFromTop) :
  s.rankFromBottom = 49 := by
  sorry

#check rank_from_bottom_calculation

end rank_from_bottom_calculation_l424_42480


namespace set_operations_l424_42481

def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧
  (Set.univ \ B = {x | x ≥ 6 ∨ x ≤ -6}) ∧
  (A \ B = {x | x ≥ 6}) ∧
  (A \ (A \ B) = {x | 4 < x ∧ x < 6}) :=
by sorry

end set_operations_l424_42481


namespace tangent_line_at_one_l424_42421

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x - 1/x) - 2 * Real.log x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 + 2/(x^2) - 2/x

-- Theorem statement
theorem tangent_line_at_one (x y : ℝ) :
  (y = f x) → (x = 1) → (2*x - y - 2 = 0) :=
sorry

end

end tangent_line_at_one_l424_42421


namespace symmetry_implies_line_equation_l424_42492

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (circle1 circle2 line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 ∧ 
  (∃ (x y : ℝ), line x y ∧ 
    ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) ∧
    ((x1 + x2) / 2 = x) ∧ ((y1 + y2) / 2 = y))

-- Theorem statement
theorem symmetry_implies_line_equation : 
  symmetric_wrt_line circle_O circle_C line_l :=
sorry

end symmetry_implies_line_equation_l424_42492


namespace min_value_of_expression_l424_42475

theorem min_value_of_expression (x y : ℝ) : (2*x*y - 3)^2 + (x - y)^2 ≥ 1 := by
  sorry

end min_value_of_expression_l424_42475


namespace tom_running_distance_l424_42425

def base_twelve_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 12^3 + ((n / 100) % 10) * 12^2 + ((n / 10) % 10) * 12^1 + (n % 10)

def average_per_week (total : ℕ) (weeks : ℕ) : ℚ :=
  (total : ℚ) / (weeks : ℚ)

theorem tom_running_distance :
  let base_twelve_distance : ℕ := 3847
  let decimal_distance : ℕ := base_twelve_to_decimal base_twelve_distance
  let weeks : ℕ := 4
  decimal_distance = 6391 ∧ average_per_week decimal_distance weeks = 1597.75 := by
  sorry

end tom_running_distance_l424_42425


namespace speech_competition_selection_l424_42419

def total_students : Nat := 9
def num_boys : Nat := 5
def num_girls : Nat := 4
def students_to_select : Nat := 4

def selection_methods : Nat := sorry

theorem speech_competition_selection :
  (total_students = num_boys + num_girls) →
  (students_to_select ≤ total_students) →
  (selection_methods = 86) := by sorry

end speech_competition_selection_l424_42419


namespace locus_is_ellipse_l424_42472

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 64

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define the point Q (center of the given circle)
def Q : ℝ × ℝ := (6, 0)

-- Define a circle passing through P and tangent to the given circle
def passingCircle (a b r : ℝ) : Prop :=
  (a - P.1)^2 + (b - P.2)^2 = r^2 ∧
  ∃ (x y : ℝ), givenCircle x y ∧ (a - x)^2 + (b - y)^2 = r^2 ∧
  (a - Q.1)^2 + (b - Q.2)^2 = (8 - r)^2

-- Define the locus of centers
def locus (a b : ℝ) : Prop :=
  ∃ (r : ℝ), passingCircle a b r

-- Theorem statement
theorem locus_is_ellipse :
  ∀ (a b : ℝ), locus a b ↔ 
    (a - P.1)^2 + (b - P.2)^2 + (a - Q.1)^2 + (b - Q.2)^2 = 8^2 :=
sorry

end locus_is_ellipse_l424_42472


namespace least_α_is_correct_l424_42412

/-- An isosceles triangle with two equal angles α° and a third angle β° -/
structure IsoscelesTriangle where
  α : ℕ
  β : ℕ
  is_isosceles : α + α + β = 180
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β
  α_ne_β : α ≠ β

/-- The least possible value of α in an isosceles triangle where α and β are distinct primes -/
def least_α : ℕ := 41

theorem least_α_is_correct (t : IsoscelesTriangle) : t.α ≥ least_α := by
  sorry

end least_α_is_correct_l424_42412


namespace h3po4_naoh_reaction_results_l424_42424

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a balanced chemical equation --/
structure BalancedEquation where
  reactant1 : Compound
  reactant2 : Compound
  product1 : Compound
  product2 : Compound
  stoichiometry : ℝ

/-- Determines the limiting reactant and calculates reaction results --/
def reactionResults (eq : BalancedEquation) : Compound × Compound × Compound := sorry

/-- Theorem stating the reaction results for H3PO4 and NaOH --/
theorem h3po4_naoh_reaction_results :
  let h3po4 := Compound.mk "H3PO4" 2.5
  let naoh := Compound.mk "NaOH" 3
  let equation := BalancedEquation.mk h3po4 naoh (Compound.mk "Na3PO4" 0) (Compound.mk "H2O" 0) 3
  let (h2o_formed, limiting_reactant, unreacted_h3po4) := reactionResults equation
  h2o_formed.moles = 3 ∧
  limiting_reactant.name = "NaOH" ∧
  unreacted_h3po4.moles = 1.5 := by sorry

end h3po4_naoh_reaction_results_l424_42424


namespace tangents_divide_plane_l424_42485

/-- The number of regions created by n lines in a plane --/
def num_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => num_regions k + (k + 1)

/-- Theorem: 7 distinct tangents to a circle divide the plane into 29 regions --/
theorem tangents_divide_plane : num_regions 7 = 29 := by
  sorry

/-- Lemma: The number of regions for n tangents follows the recursive formula R(n) = R(n-1) + n --/
lemma regions_recursive_formula (n : ℕ) : num_regions (n + 1) = num_regions n + (n + 1) := by
  sorry

end tangents_divide_plane_l424_42485


namespace radio_profit_percentage_is_approximately_6_8_percent_l424_42413

/-- Calculates the profit percentage for a radio sale given the following parameters:
    * initial_cost: The initial cost of the radio
    * overhead: Overhead expenses
    * purchase_tax_rate: Purchase tax rate
    * luxury_tax_rate: Luxury tax rate
    * exchange_discount_rate: Exchange offer discount rate
    * sales_tax_rate: Sales tax rate
    * selling_price: Final selling price
-/
def calculate_profit_percentage (
  initial_cost : ℝ
  ) (overhead : ℝ
  ) (purchase_tax_rate : ℝ
  ) (luxury_tax_rate : ℝ
  ) (exchange_discount_rate : ℝ
  ) (sales_tax_rate : ℝ
  ) (selling_price : ℝ
  ) : ℝ :=
  sorry

/-- The profit percentage for the radio sale is approximately 6.8% -/
theorem radio_profit_percentage_is_approximately_6_8_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_profit_percentage 225 28 0.08 0.05 0.10 0.12 300 - 6.8| < ε :=
sorry

end radio_profit_percentage_is_approximately_6_8_percent_l424_42413


namespace student_fraction_mistake_l424_42464

theorem student_fraction_mistake (n : ℚ) (correct_fraction : ℚ) (student_fraction : ℚ) :
  n = 288 →
  correct_fraction = 5 / 16 →
  student_fraction * n = correct_fraction * n + 150 →
  student_fraction = 5 / 6 := by
sorry

end student_fraction_mistake_l424_42464


namespace arithmetic_calculation_l424_42495

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_calculation_l424_42495


namespace smallest_prime_divisor_of_sum_l424_42417

theorem smallest_prime_divisor_of_sum : ∃ k : ℕ, 4^15 + 6^17 = 2 * k := by
  sorry

end smallest_prime_divisor_of_sum_l424_42417


namespace find_N_l424_42467

theorem find_N : ∃ N : ℕ+, (22 ^ 2 * 55 ^ 2 : ℕ) = 10 ^ 2 * N ^ 2 ∧ N = 121 := by
  sorry

end find_N_l424_42467


namespace triangle_area_234_l424_42489

theorem triangle_area_234 : 
  let a := 2
  let b := 3
  let c := 4
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = (3 * Real.sqrt 15) / 4 := by
sorry

end triangle_area_234_l424_42489


namespace curve_intersects_all_planes_l424_42445

/-- A smooth curve in ℝ³ -/
def C : ℝ → ℝ × ℝ × ℝ := fun t ↦ (t, t^3, t^5)

/-- Definition of a plane in ℝ³ -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- The theorem stating that the curve C intersects every plane -/
theorem curve_intersects_all_planes :
  ∀ (p : Plane), ∃ (t : ℝ), 
    let (x, y, z) := C t
    p.A * x + p.B * y + p.C * z + p.D = 0 := by
  sorry


end curve_intersects_all_planes_l424_42445


namespace min_score_for_average_l424_42416

def total_tests : ℕ := 7
def max_score : ℕ := 100
def target_average : ℕ := 80

def first_four_scores : List ℕ := [82, 90, 78, 85]

theorem min_score_for_average (scores : List ℕ) 
  (h1 : scores.length = 4)
  (h2 : ∀ s ∈ scores, s ≤ max_score) :
  ∃ (x y z : ℕ),
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / total_tests = target_average ∧
    (∀ a b c : ℕ, 
      a ≤ max_score → b ≤ max_score → c ≤ max_score →
      (scores.sum + a + b + c) / total_tests = target_average →
      min x y ≤ min a b ∧ min x y ≤ c) ∧
    25 = min x (min y z) := by
  sorry

#check min_score_for_average first_four_scores

end min_score_for_average_l424_42416


namespace complement_union_problem_l424_42435

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-2, 2}

theorem complement_union_problem : (U \ A) ∪ B = {-2, 0, 1, 2} := by
  sorry

end complement_union_problem_l424_42435


namespace average_visitors_per_day_l424_42499

def visitor_counts : List ℕ := [583, 246, 735, 492, 639]
def num_days : ℕ := 5

theorem average_visitors_per_day :
  (visitor_counts.sum / num_days : ℚ) = 539 := by
  sorry

end average_visitors_per_day_l424_42499


namespace range_of_m_l424_42451

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m - 4 = 0 ∧ 
              x₂^2 - x₂ + m - 4 = 0 ∧ 
              x₁ * x₂ < 0

-- Main theorem
theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by sorry

end range_of_m_l424_42451


namespace complex_modulus_problem_l424_42477

theorem complex_modulus_problem : 
  Complex.abs ((1 + Complex.I) * (2 - Complex.I)) = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l424_42477


namespace money_distribution_l424_42436

/-- Given a total amount of money and the fraction one person has relative to the others,
    calculate how much money that person has. -/
theorem money_distribution (total : ℕ) (fraction : ℚ) (person_amount : ℕ) : 
  total = 7000 →
  fraction = 2 / 3 →
  person_amount = total * (fraction / (1 + fraction)) →
  person_amount = 2800 := by
  sorry

#check money_distribution

end money_distribution_l424_42436


namespace pauls_vertical_distance_l424_42468

/-- The number of feet Paul travels vertically in a week -/
def vertical_distance_per_week (story : ℕ) (trips_per_day : ℕ) (days_per_week : ℕ) (feet_per_story : ℕ) : ℕ :=
  2 * story * trips_per_day * days_per_week * feet_per_story

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  vertical_distance_per_week 5 3 7 10 = 2100 := by
  sorry

#eval vertical_distance_per_week 5 3 7 10

end pauls_vertical_distance_l424_42468


namespace reciprocals_inversely_proportional_l424_42441

/-- Two real numbers are inversely proportional if their product is constant --/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

/-- Theorem: If x and y are inversely proportional, then their reciprocals are also inversely proportional --/
theorem reciprocals_inversely_proportional
  (x y : ℝ → ℝ)
  (h : InverselyProportional x y)
  (hx : ∀ t, x t ≠ 0)
  (hy : ∀ t, y t ≠ 0) :
  InverselyProportional (fun t ↦ 1 / x t) (fun t ↦ 1 / y t) :=
by
  sorry

end reciprocals_inversely_proportional_l424_42441


namespace quadratic_inequality_equivalence_l424_42460

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) ↔ a ≥ 2 :=
sorry

end quadratic_inequality_equivalence_l424_42460


namespace perpendicular_parallel_implies_perpendicular_l424_42476

/-- A type representing lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A type representing planes in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem perpendicular_parallel_implies_perpendicular 
  (b c : Line3D) (α : Plane3D) :
  perpendicular_line_plane b α → 
  parallel_line_plane c α → 
  perpendicular_lines b c :=
sorry

end perpendicular_parallel_implies_perpendicular_l424_42476
