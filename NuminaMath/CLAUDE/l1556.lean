import Mathlib

namespace coltons_remaining_stickers_l1556_155630

/-- The number of stickers Colton has left after giving some to his friends. -/
def stickers_left (initial : ℕ) (per_friend : ℕ) (num_friends : ℕ) (mandy_extra : ℕ) (justin_less : ℕ) : ℕ :=
  let friends_total := per_friend * num_friends
  let mandy_stickers := friends_total + mandy_extra
  let justin_stickers := mandy_stickers - justin_less
  initial - (friends_total + mandy_stickers + justin_stickers)

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers :
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end coltons_remaining_stickers_l1556_155630


namespace expression_factorization_l1556_155681

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end expression_factorization_l1556_155681


namespace g_eval_sqrt_half_l1556_155665

noncomputable def g (x : ℝ) : ℝ := Real.arccos (x^2) * Real.arcsin (x^2)

theorem g_eval_sqrt_half : g (1 / Real.sqrt 2) = π^2 / 18 := by
  sorry

end g_eval_sqrt_half_l1556_155665


namespace ellipse_b_value_l1556_155660

/-- The value of b for an ellipse with given properties -/
theorem ellipse_b_value (b : ℝ) (h1 : 0 < b) (h2 : b < 3) :
  (∀ x y : ℝ, x^2 / 9 + y^2 / b^2 = 1 →
    ∃ F₁ F₂ : ℝ × ℝ, 
      (F₁.1 < 0 ∧ F₂.1 > 0) ∧ 
      (∀ A B : ℝ × ℝ, 
        (A.1^2 / 9 + A.2^2 / b^2 = 1) ∧ 
        (B.1^2 / 9 + B.2^2 / b^2 = 1) ∧
        (∃ k : ℝ, A.2 = k * (A.1 - F₁.1) ∧ B.2 = k * (B.1 - F₁.1)) →
        (dist A F₁ + dist A F₂ = 6) ∧ 
        (dist B F₁ + dist B F₂ = 6) ∧
        (dist B F₂ + dist A F₂ ≤ 10))) →
  b = Real.sqrt 3 :=
sorry

end ellipse_b_value_l1556_155660


namespace given_segments_proportionate_l1556_155668

/-- A set of line segments is proportionate if the product of any two segments
    equals the product of the remaining two segments. -/
def IsProportionate (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The given set of line segments. -/
def LineSegments : (ℝ × ℝ × ℝ × ℝ) :=
  (3, 6, 4, 8)

/-- Theorem stating that the given set of line segments is proportionate. -/
theorem given_segments_proportionate :
  let (a, b, c, d) := LineSegments
  IsProportionate a b c d := by
  sorry

end given_segments_proportionate_l1556_155668


namespace total_oranges_bought_l1556_155698

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- Theorem: The total number of oranges Stephanie bought last month is 16 -/
theorem total_oranges_bought : store_visits * oranges_per_visit = 16 := by
  sorry

end total_oranges_bought_l1556_155698


namespace addition_problem_l1556_155684

theorem addition_problem : ∃ x : ℝ, 37 + x = 52 ∧ x = 15 := by
  sorry

end addition_problem_l1556_155684


namespace fourth_grade_agreement_l1556_155638

theorem fourth_grade_agreement (third_grade : ℕ) (total : ℕ) (h1 : third_grade = 154) (h2 : total = 391) :
  total - third_grade = 237 := by
  sorry

end fourth_grade_agreement_l1556_155638


namespace painted_subcubes_count_l1556_155612

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  all_faces_painted : Bool

/-- Calculates the number of 1x1x1 subcubes with at least two painted faces in a painted cube -/
def subcubes_with_two_or_more_painted_faces (c : Cube 4) : ℕ :=
  if c.all_faces_painted then
    -- Corner cubes (3 faces painted)
    8 +
    -- Edge cubes without corners (2 faces painted)
    (12 * 2) +
    -- Middle-edge face cubes (2 faces painted)
    (6 * 4)
  else
    0

/-- Theorem: In a 4x4x4 cube with all faces painted, there are 56 subcubes with at least two painted faces -/
theorem painted_subcubes_count (c : Cube 4) (h : c.all_faces_painted = true) :
  subcubes_with_two_or_more_painted_faces c = 56 := by
  sorry

#check painted_subcubes_count

end painted_subcubes_count_l1556_155612


namespace parabola_focus_directrix_distance_l1556_155694

/-- A parabola with vertex at the origin and focus on the x-axis. -/
structure Parabola where
  /-- The x-coordinate of the focus -/
  p : ℝ
  /-- The parabola passes through this point -/
  point : ℝ × ℝ

/-- The distance from the focus to the directrix for a parabola -/
def focusDirectrixDistance (c : Parabola) : ℝ :=
  c.p

theorem parabola_focus_directrix_distance 
  (c : Parabola) 
  (h1 : c.point = (1, 3)) : 
  focusDirectrixDistance c = 9/2 := by
  sorry

#check parabola_focus_directrix_distance

end parabola_focus_directrix_distance_l1556_155694


namespace cos_2theta_value_l1556_155634

theorem cos_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) :
  Real.cos (2 * θ) = -1/3 := by
sorry

end cos_2theta_value_l1556_155634


namespace expression_evaluation_l1556_155666

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1/2
  2 * (3 * x^3 - x + 3 * y) - (x - 2 * y + 6 * x^3) = -3 :=
by sorry

end expression_evaluation_l1556_155666


namespace ivory_josh_riddle_difference_l1556_155625

theorem ivory_josh_riddle_difference :
  ∀ (ivory_riddles josh_riddles taso_riddles : ℕ),
    josh_riddles = 8 →
    taso_riddles = 24 →
    taso_riddles = 2 * ivory_riddles →
    ivory_riddles - josh_riddles = 4 :=
by
  sorry

end ivory_josh_riddle_difference_l1556_155625


namespace min_value_cos_half_theta_times_two_minus_sin_theta_l1556_155664

theorem min_value_cos_half_theta_times_two_minus_sin_theta (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (min : Real), min = 0 ∧ ∀ θ', 0 < θ' ∧ θ' < π →
    min ≤ Real.cos (θ' / 2) * (2 - Real.sin θ') :=
by sorry

end min_value_cos_half_theta_times_two_minus_sin_theta_l1556_155664


namespace watch_synchronization_l1556_155607

/-- The number of seconds in a full rotation of a standard watch -/
def full_rotation : ℕ := 12 * 60 * 60

/-- The number of seconds Glafira's watch gains per day -/
def glafira_gain : ℕ := 12

/-- The number of seconds Gavrila's watch loses per day -/
def gavrila_loss : ℕ := 18

/-- The combined deviation of both watches per day -/
def combined_deviation : ℕ := glafira_gain + gavrila_loss

theorem watch_synchronization :
  (full_rotation / combined_deviation : ℚ) = 1440 := by sorry

end watch_synchronization_l1556_155607


namespace unique_phone_number_l1556_155675

def is_valid_phone_number (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000

def first_four (n : ℕ) : ℕ := n / 10000

def last_four (n : ℕ) : ℕ := n % 10000

def first_three (n : ℕ) : ℕ := n / 100000

def last_five (n : ℕ) : ℕ := n % 100000

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧
    first_four n + last_four n = 14405 ∧
    first_three n + last_five n = 16970 ∧
    n = 82616144 := by
  sorry

end unique_phone_number_l1556_155675


namespace linear_function_triangle_area_l1556_155600

theorem linear_function_triangle_area (k : ℝ) : 
  (1/2 * 3 * |3/k| = 24) → (k = 3/16 ∨ k = -3/16) := by
  sorry

end linear_function_triangle_area_l1556_155600


namespace sum_y_invariant_under_rotation_l1556_155677

/-- A rectangle in 2D space -/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  is_opposite : v1 ≠ v2

/-- The sum of y-coordinates of two points -/
def sum_y (p1 p2 : ℝ × ℝ) : ℝ := p1.2 + p2.2

/-- Theorem: The sum of y-coordinates of the other two vertices of a rectangle
    remains unchanged after a 90-degree rotation around its center -/
theorem sum_y_invariant_under_rotation (r : Rectangle) 
    (h1 : r.v1 = (5, 20))
    (h2 : r.v2 = (11, -8)) :
    ∃ (v3 v4 : ℝ × ℝ), sum_y v3 v4 = 12 ∧ 
    (∀ (v3' v4' : ℝ × ℝ), sum_y v3' v4' = 12) :=
  sorry

#check sum_y_invariant_under_rotation

end sum_y_invariant_under_rotation_l1556_155677


namespace triangle_area_with_given_conditions_l1556_155647

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 2, R = 9, and 2cos(E) = cos(D) + cos(F), then the area of the triangle is 54. -/
theorem triangle_area_with_given_conditions (D E F : Real) (r R : Real) :
  r = 2 →
  R = 9 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (area : Real), area = 54 ∧ area = r * (Real.sin D + Real.sin E + Real.sin F) * R / 2 := by
  sorry


end triangle_area_with_given_conditions_l1556_155647


namespace unique_digit_solution_l1556_155674

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem unique_digit_solution (A M C : ℕ) 
  (h_A : is_digit A) (h_M : is_digit M) (h_C : is_digit C)
  (h_eq : (100*A + 10*M + C) * (A + M + C) = 2005) : 
  A = 4 := by
sorry

end unique_digit_solution_l1556_155674


namespace hotel_room_number_contradiction_l1556_155633

theorem hotel_room_number_contradiction : 
  ¬ ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    100 * a + 10 * b + c = (a + 1) * (b + 1) * c :=
by sorry

end hotel_room_number_contradiction_l1556_155633


namespace marathon_average_time_l1556_155685

theorem marathon_average_time (casey_time : ℝ) (zendaya_factor : ℝ) : 
  casey_time = 6 → 
  zendaya_factor = 1/3 → 
  let zendaya_time := casey_time + zendaya_factor * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
sorry

end marathon_average_time_l1556_155685


namespace sum_of_coefficients_l1556_155692

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^10 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1023 :=
by
  sorry

end sum_of_coefficients_l1556_155692


namespace quadratic_inequality_solution_l1556_155611

theorem quadratic_inequality_solution (p q : ℝ) :
  (∀ x, (1/p) * x^2 + q * x + p > 0 ↔ 2 < x ∧ x < 4) →
  p = -2 * Real.sqrt 2 ∧ q = (3/2) * Real.sqrt 2 := by
sorry

end quadratic_inequality_solution_l1556_155611


namespace combined_salaries_l1556_155623

/-- Given the salary of A and the average salary of A, B, C, D, and E,
    prove the combined salaries of B, C, D, and E. -/
theorem combined_salaries
  (salary_A : ℕ)
  (average_salary : ℕ)
  (h1 : salary_A = 10000)
  (h2 : average_salary = 8400) :
  salary_A + (4 * ((5 * average_salary) - salary_A)) = 42000 :=
by sorry

end combined_salaries_l1556_155623


namespace f_properties_l1556_155639

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

theorem f_properties :
  (∀ a : ℝ, (∀ x > 0, x * (Real.log x + 1/x) ≤ x^2 + a*x + 1) ↔ a ≥ -1) ∧
  (∀ x > 0, (x - 1) * f x ≥ 0) := by
  sorry

end f_properties_l1556_155639


namespace divisibility_by_900_l1556_155656

theorem divisibility_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end divisibility_by_900_l1556_155656


namespace ride_time_is_36_seconds_l1556_155662

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  theo_walk_time_non_operating : ℝ
  theo_walk_time_operating : ℝ
  escalator_efficiency : ℝ
  theo_walk_time_non_operating_eq : theo_walk_time_non_operating = 80
  theo_walk_time_operating_eq : theo_walk_time_operating = 30
  escalator_efficiency_eq : escalator_efficiency = 0.75

/-- Calculates the time it takes Theo to ride down the operating escalator while standing still -/
def ride_time (problem : EscalatorProblem) : ℝ :=
  problem.theo_walk_time_non_operating * problem.escalator_efficiency

/-- Theorem stating that the ride time for Theo is 36 seconds -/
theorem ride_time_is_36_seconds (problem : EscalatorProblem) :
  ride_time problem = 36 := by
  sorry

#eval ride_time { theo_walk_time_non_operating := 80,
                  theo_walk_time_operating := 30,
                  escalator_efficiency := 0.75,
                  theo_walk_time_non_operating_eq := rfl,
                  theo_walk_time_operating_eq := rfl,
                  escalator_efficiency_eq := rfl }

end ride_time_is_36_seconds_l1556_155662


namespace problem_solution_l1556_155636

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 1| - |x - a|

def g (a : ℝ) (x : ℝ) : ℝ := f a x + 3 * |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0) ∧
  (∃ t : ℝ, (∀ x : ℝ, g 1 x ≥ t) ∧
   (∀ m n : ℝ, m > 0 → n > 0 → 2/m + 1/(2*n) = t →
    m + n ≥ 9/8) ∧
   (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2/m + 1/(2*n) = t ∧ m + n = 9/8)) :=
by sorry

end

end problem_solution_l1556_155636


namespace compound_interest_problem_l1556_155653

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8000 ∧
  P * (1 + r)^3 = 9261 :=
by sorry

end compound_interest_problem_l1556_155653


namespace difference_of_squares_302_298_l1556_155601

theorem difference_of_squares_302_298 : 302^2 - 298^2 = 2400 := by
  sorry

end difference_of_squares_302_298_l1556_155601


namespace andrei_apple_spending_l1556_155680

/-- Calculates Andrei's monthly spending on apples after price increase and discount -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let discountedPrice := newPrice * (1 - discount)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 (1/10) (1/10) 2 = 99 := by
  sorry

end andrei_apple_spending_l1556_155680


namespace unique_solution_quadratic_l1556_155641

theorem unique_solution_quadratic (n : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = n + 3 * x) ↔ n = 35 / 4 := by
  sorry

end unique_solution_quadratic_l1556_155641


namespace x_y_not_congruent_l1556_155619

def x : ℕ → ℕ
  | 0 => 365
  | n + 1 => x n * (x n ^ 1986 + 1) + 1622

def y : ℕ → ℕ
  | 0 => 16
  | n + 1 => y n * (y n ^ 3 + 1) - 1952

theorem x_y_not_congruent (n k : ℕ) : x n % 1987 ≠ y k % 1987 := by
  sorry

end x_y_not_congruent_l1556_155619


namespace florist_roses_count_l1556_155672

/-- Calculates the final number of roses a florist has after selling and picking more. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that given the initial conditions, the florist ends up with 56 roses. -/
theorem florist_roses_count : final_roses 50 15 21 = 56 := by
  sorry

end florist_roses_count_l1556_155672


namespace chicken_ratio_is_two_to_one_l1556_155609

/-- The number of chickens in the coop -/
def chickens_in_coop : ℕ := 14

/-- The number of chickens free ranging -/
def chickens_free_ranging : ℕ := 52

/-- The number of chickens in the run -/
def chickens_in_run : ℕ := (chickens_free_ranging + 4) / 2

/-- The ratio of chickens in the run to chickens in the coop -/
def chicken_ratio : ℚ := chickens_in_run / chickens_in_coop

theorem chicken_ratio_is_two_to_one : chicken_ratio = 2 := by
  sorry

end chicken_ratio_is_two_to_one_l1556_155609


namespace statement_a_is_false_l1556_155616

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  isotopes : List (ℕ × ℝ)  -- List of (mass number, abundance) pairs

/-- Calculates the relative atomic mass of an element -/
def relative_atomic_mass (e : Element) : ℝ :=
  (e.isotopes.map (λ (mass, abundance) => mass * abundance)).sum

/-- Represents a single atom of an element -/
structure Atom where
  protons : ℕ
  neutrons : ℕ

/-- The statement we want to prove false -/
def statement_a (e : Element) (a : Atom) : Prop :=
  relative_atomic_mass e = a.protons + a.neutrons

/-- Theorem stating that the statement is false -/
theorem statement_a_is_false :
  ∃ (e : Element) (a : Atom), ¬(statement_a e a) :=
sorry

end statement_a_is_false_l1556_155616


namespace vectors_orthogonal_l1556_155606

theorem vectors_orthogonal (x : ℝ) : 
  x = 28/3 → (3 * x + 4 * (-7) = 0) := by sorry

end vectors_orthogonal_l1556_155606


namespace system_solution_l1556_155628

theorem system_solution (x y z u : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * z + y * u = 7)
  (eq3 : x * z^2 + y * u^2 = 12)
  (eq4 : x * z^3 + y * u^3 = 21) :
  z = 7/3 ∧ y = -Real.sqrt 3 := by
sorry

end system_solution_l1556_155628


namespace bathroom_tiling_savings_janet_bathroom_savings_l1556_155652

/-- Calculates the savings when choosing the least expensive tiles over the most expensive ones for a bathroom tiling project. -/
theorem bathroom_tiling_savings (wall1_length wall1_width wall2_length wall2_width wall3_length wall3_width : ℕ)
  (tiles_per_sqft : ℕ) (cheap_tile_cost expensive_tile_cost : ℚ) : ℚ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width + wall3_length * wall3_width
  let total_tiles := total_area * tiles_per_sqft
  let expensive_total := total_tiles * expensive_tile_cost
  let cheap_total := total_tiles * cheap_tile_cost
  expensive_total - cheap_total

/-- The savings for Janet's specific bathroom tiling project is $2,400. -/
theorem janet_bathroom_savings : 
  bathroom_tiling_savings 5 8 7 8 6 9 4 11 15 = 2400 := by
  sorry

end bathroom_tiling_savings_janet_bathroom_savings_l1556_155652


namespace chromosome_set_variation_l1556_155624

/-- Represents the types of chromosome number variations -/
inductive ChromosomeVariationType
| IndividualChange
| SetChange

/-- Represents the form of chromosome changes -/
inductive ChromosomeChangeForm
| Individual
| Set

/-- Definition of chromosome number variation -/
structure ChromosomeVariation where
  type : ChromosomeVariationType
  form : ChromosomeChangeForm

/-- Theorem stating that one type of chromosome number variation involves
    doubling or halving of chromosomes in the form of chromosome sets -/
theorem chromosome_set_variation :
  ∃ (cv : ChromosomeVariation),
    cv.type = ChromosomeVariationType.SetChange ∧
    cv.form = ChromosomeChangeForm.Set :=
sorry

end chromosome_set_variation_l1556_155624


namespace factorization_of_polynomial_l1556_155642

theorem factorization_of_polynomial (x : ℝ) : 
  x^4 - 3*x^3 - 28*x^2 = x^2 * (x - 7) * (x + 4) := by
  sorry

end factorization_of_polynomial_l1556_155642


namespace crayon_distribution_l1556_155627

theorem crayon_distribution (total benny fred jason sarah : ℕ) : 
  total = 96 →
  benny = 12 →
  fred = 2 * benny →
  jason = 3 * sarah →
  fred + benny + jason + sarah = total →
  (fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15) :=
by sorry

end crayon_distribution_l1556_155627


namespace factorization_x_squared_minus_2023x_l1556_155604

theorem factorization_x_squared_minus_2023x (x : ℝ) : x^2 - 2023*x = x*(x - 2023) := by
  sorry

end factorization_x_squared_minus_2023x_l1556_155604


namespace marbles_lost_vs_found_l1556_155620

theorem marbles_lost_vs_found (initial : ℕ) (lost : ℕ) (found : ℕ) : 
  initial = 4 → lost = 16 → found = 8 → lost - found = 8 := by
  sorry

end marbles_lost_vs_found_l1556_155620


namespace exists_special_function_l1556_155691

/-- The number of divisors of a natural number -/
def number_of_divisors (n : ℕ) : ℕ := sorry

/-- The existence of a function with specific properties -/
theorem exists_special_function : 
  ∃ f : ℕ → ℕ, 
    (∃ n : ℕ, f n ≠ n) ∧ 
    (∀ m n : ℕ, (number_of_divisors m = f n) ↔ (number_of_divisors (f m) = n)) :=
sorry

end exists_special_function_l1556_155691


namespace randys_trip_length_l1556_155644

theorem randys_trip_length :
  ∀ (total_length : ℚ),
    (1 / 3 : ℚ) * total_length + 20 + (1 / 5 : ℚ) * total_length = total_length →
    total_length = 300 / 7 := by
  sorry

end randys_trip_length_l1556_155644


namespace smallest_positive_d_for_inequality_l1556_155602

theorem smallest_positive_d_for_inequality :
  (∃ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|)) ∧
  (∀ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|) →
    d ≥ 1) :=
by sorry

end smallest_positive_d_for_inequality_l1556_155602


namespace fourth_term_coefficient_specific_case_l1556_155621

def binomial_expansion (a b : ℝ) (n : ℕ) := (a + b)^n

def fourth_term_coefficient (a b : ℝ) (n : ℕ) : ℝ :=
  Nat.choose n 3 * a^(n-3) * b^3

theorem fourth_term_coefficient_specific_case :
  fourth_term_coefficient (1/2 * Real.sqrt x) (2/(3*x)) 6 = 20 :=
by
  sorry

end fourth_term_coefficient_specific_case_l1556_155621


namespace orange_harvest_calculation_l1556_155657

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks harvested after the given number of days -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_calculation :
  total_sacks = 1862 := by sorry

end orange_harvest_calculation_l1556_155657


namespace jason_seashells_l1556_155697

/-- The number of seashells Jason has now -/
def current_seashells : ℕ := 36

/-- The number of seashells Jason gave away -/
def given_away_seashells : ℕ := 13

/-- The initial number of seashells Jason found -/
def initial_seashells : ℕ := current_seashells + given_away_seashells

theorem jason_seashells : initial_seashells = 49 := by
  sorry

end jason_seashells_l1556_155697


namespace inequality_proof_l1556_155679

theorem inequality_proof (α β γ : ℝ) 
  (h1 : β * γ ≠ 0) 
  (h2 : (1 - γ^2) / (β * γ) ≥ 0) : 
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ := by
  sorry

end inequality_proof_l1556_155679


namespace complex_expression_evaluation_l1556_155695

theorem complex_expression_evaluation :
  let a : ℂ := 1 + 2*I
  let b : ℂ := 2 + I
  a * b - 2 * b^2 = -6 - 3*I :=
by sorry

end complex_expression_evaluation_l1556_155695


namespace special_function_value_l1556_155654

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that f(2010) = 2 for any function satisfying the conditions -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2010 = 2 := by
  sorry


end special_function_value_l1556_155654


namespace cranberry_juice_price_per_ounce_l1556_155650

/-- Given a can of cranberry juice with volume in ounces and price in cents,
    calculate the price per ounce in cents. -/
def price_per_ounce (volume : ℕ) (price : ℕ) : ℚ :=
  price / volume

/-- Theorem stating that the price per ounce of cranberry juice is 7 cents
    given that a 12 ounce can sells for 84 cents. -/
theorem cranberry_juice_price_per_ounce :
  price_per_ounce 12 84 = 7 := by
  sorry

end cranberry_juice_price_per_ounce_l1556_155650


namespace robin_gum_total_l1556_155635

/-- Calculate the total number of gum pieces Robin has after his purchases -/
theorem robin_gum_total (initial_packages : ℕ) (initial_pieces_per_package : ℕ)
  (local_packages : ℚ) (local_pieces_per_package : ℕ)
  (foreign_packages : ℕ) (foreign_pieces_per_package : ℕ)
  (exchange_rate : ℚ) (foreign_purchase_dollars : ℕ) :
  initial_packages = 27 →
  initial_pieces_per_package = 18 →
  local_packages = 15.5 →
  local_pieces_per_package = 12 →
  foreign_packages = 8 →
  foreign_pieces_per_package = 25 →
  exchange_rate = 1.2 →
  foreign_purchase_dollars = 50 →
  (initial_packages * initial_pieces_per_package +
   ⌊local_packages⌋ * local_pieces_per_package +
   foreign_packages * foreign_pieces_per_package) = 872 := by
  sorry

#check robin_gum_total

end robin_gum_total_l1556_155635


namespace gcd_91_72_l1556_155670

theorem gcd_91_72 : Nat.gcd 91 72 = 1 := by
  sorry

end gcd_91_72_l1556_155670


namespace complex_division_result_l1556_155608

theorem complex_division_result : (4 + 3*I : ℂ) / (2 - I) = 1 + 2*I := by sorry

end complex_division_result_l1556_155608


namespace talia_father_age_l1556_155699

/-- Represents the ages of Talia and her parents -/
structure FamilyAges where
  talia : ℕ
  mom : ℕ
  dad : ℕ

/-- Conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.talia + 7 = 20 ∧
  ages.mom = 3 * ages.talia ∧
  ages.dad + 3 = ages.mom

/-- Theorem stating that Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  problem_conditions ages → ages.dad = 36 := by
  sorry


end talia_father_age_l1556_155699


namespace square_root_calculation_l1556_155690

theorem square_root_calculation : Real.sqrt (5^2 - 4^2 - 3^2) = 0 := by
  sorry

end square_root_calculation_l1556_155690


namespace quadratic_polynomial_determination_l1556_155632

/-- A type representing quadratic polynomials -/
def QuadraticPolynomial := ℝ → ℝ

/-- A function that evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ := p x

/-- A function that checks if two polynomials agree at a given point -/
def agree (p q : QuadraticPolynomial) (x : ℝ) : Prop := evaluate p x = evaluate q x

theorem quadratic_polynomial_determination (n : ℕ) (h : n > 1) :
  ∃ (C : ℝ), C > 0 ∧
  ∀ (polynomials : Finset QuadraticPolynomial),
  polynomials.card = n →
  ∃ (points : Finset ℝ),
  points.card = 2 * n^2 + 1 ∧
  ∃ (p : QuadraticPolynomial),
  p ∈ polynomials ∧
  ∀ (q : QuadraticPolynomial),
  (∀ (x : ℝ), x ∈ points → agree p q x) → p = q :=
sorry

end quadratic_polynomial_determination_l1556_155632


namespace roots_of_equation_l1556_155659

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun y ↦ (2 * y + 1) * (2 * y - 3)
  ∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/2 ∧ (∀ y : ℝ, f y = 0 ↔ y = y₁ ∨ y = y₂) := by
  sorry

end roots_of_equation_l1556_155659


namespace distance_from_center_to_point_l1556_155678

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 18

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the point
def point : ℝ × ℝ := (3, -2)

-- Theorem statement
theorem distance_from_center_to_point : 
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 2 :=
by sorry

end distance_from_center_to_point_l1556_155678


namespace john_walks_farther_l1556_155646

/-- John's walking distance to school in miles -/
def john_distance : ℝ := 1.74

/-- Nina's walking distance to school in miles -/
def nina_distance : ℝ := 1.235

/-- The difference between John's and Nina's walking distances -/
def distance_difference : ℝ := john_distance - nina_distance

theorem john_walks_farther : distance_difference = 0.505 := by sorry

end john_walks_farther_l1556_155646


namespace new_ratio_is_13_to_7_l1556_155663

/-- Represents the farm's animal count before and after the transaction -/
structure FarmCount where
  initialHorses : ℕ
  initialCows : ℕ
  finalHorses : ℕ
  finalCows : ℕ

/-- Checks if the given FarmCount satisfies the problem conditions -/
def validFarmCount (f : FarmCount) : Prop :=
  f.initialHorses = 4 * f.initialCows ∧
  f.finalHorses = f.initialHorses - 15 ∧
  f.finalCows = f.initialCows + 15 ∧
  f.finalHorses = f.finalCows + 30

/-- Theorem stating that the new ratio of horses to cows is 13:7 -/
theorem new_ratio_is_13_to_7 (f : FarmCount) (h : validFarmCount f) :
  13 * f.finalCows = 7 * f.finalHorses :=
sorry

end new_ratio_is_13_to_7_l1556_155663


namespace tan_sum_reciprocal_l1556_155696

theorem tan_sum_reciprocal (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4)
  (h3 : Real.tan x * Real.tan y = 1/3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 3 := by
sorry

end tan_sum_reciprocal_l1556_155696


namespace equipment_cost_proof_l1556_155629

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 152/10

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 68/10

/-- The total cost of equipment for all players on the team -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end equipment_cost_proof_l1556_155629


namespace center_numbers_l1556_155686

def numbers : List ℕ := [9, 12, 18, 24, 36, 48, 96]

def is_valid_center (x : ℕ) (nums : List ℕ) : Prop :=
  x ∈ nums ∧
  ∃ (a b c d e f : ℕ),
    a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ e ∈ nums ∧ f ∈ nums ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    a * x * d = b * x * e ∧ b * x * e = c * x * f

theorem center_numbers :
  ∀ x ∈ numbers, is_valid_center x numbers ↔ x = 12 ∨ x = 96 := by
  sorry

end center_numbers_l1556_155686


namespace product_of_one_plus_tangents_17_and_28_l1556_155655

theorem product_of_one_plus_tangents_17_and_28 :
  (1 + Real.tan (17 * π / 180)) * (1 + Real.tan (28 * π / 180)) = 2 := by
  sorry

end product_of_one_plus_tangents_17_and_28_l1556_155655


namespace solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l1556_155605

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part 1
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Theorem for part 2
theorem range_of_m_for_all_x_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
sorry

end solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l1556_155605


namespace stating_max_correct_is_38_l1556_155688

/-- Represents the result of a multiple choice contest. -/
structure ContestResult where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- 
Calculates the maximum number of correctly answered questions in a contest.
-/
def max_correct_answers (result : ContestResult) : ℕ :=
  sorry

/-- 
Theorem stating that for the given contest parameters, 
the maximum number of correctly answered questions is 38.
-/
theorem max_correct_is_38 : 
  let result : ContestResult := {
    total_questions := 60,
    correct_points := 5,
    incorrect_points := -2,
    total_score := 150
  }
  max_correct_answers result = 38 := by
  sorry

end stating_max_correct_is_38_l1556_155688


namespace chris_money_before_birthday_l1556_155626

/-- The amount of money Chris had before his birthday -/
def money_before : ℕ := sorry

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The total amount Chris had after his birthday -/
def total_after : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before + aunt_uncle_gift + parents_gift + grandmother_gift = total_after ∧
  money_before = 159 := by sorry

end chris_money_before_birthday_l1556_155626


namespace complex_power_result_l1556_155640

theorem complex_power_result : (3 * (Complex.cos (30 * Real.pi / 180)) + 3 * Complex.I * (Complex.sin (30 * Real.pi / 180)))^4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_result_l1556_155640


namespace weaving_problem_l1556_155658

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days of weaving -/
def days : ℕ := 30

/-- Represents the amount woven on the first day -/
def first_day_production : ℚ := 5

/-- Represents the total amount of cloth woven -/
def total_production : ℚ := 390

theorem weaving_problem :
  first_day_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry

end weaving_problem_l1556_155658


namespace hiking_trip_calculation_l1556_155614

structure HikingSegment where
  distance : Float
  speed : Float

def total_distance (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance) |> List.sum

def total_time (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance / s.speed) |> List.sum

def hiking_segments : List HikingSegment := [
  { distance := 0.5, speed := 3.0 },
  { distance := 1.2, speed := 2.5 },
  { distance := 0.8, speed := 2.0 },
  { distance := 0.6, speed := 2.8 }
]

theorem hiking_trip_calculation :
  total_distance hiking_segments = 3.1 ∧
  (total_time hiking_segments * 60).round = 76 := by
  sorry

#eval total_distance hiking_segments
#eval (total_time hiking_segments * 60).round

end hiking_trip_calculation_l1556_155614


namespace sunset_time_theorem_l1556_155603

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

def time_to_12hour_format (t : Time) : Time :=
  sorry

theorem sunset_time_theorem (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 12 →
  let sunset := add_time_and_duration sunrise daylight
  let sunset_12h := time_to_12hour_format sunset
  sunset_12h.hours = 5 ∧ sunset_12h.minutes = 55 :=
sorry

end sunset_time_theorem_l1556_155603


namespace canister_capacity_ratio_l1556_155683

/-- Represents the ratio of capacities between two canisters -/
structure CanisterRatio where
  c : ℝ  -- Capacity of canister C
  d : ℝ  -- Capacity of canister D

/-- Theorem stating the ratio of canister capacities given the problem conditions -/
theorem canister_capacity_ratio (r : CanisterRatio) 
  (hc_half : r.c / 2 = r.c - (r.d / 3 - r.d / 12)) 
  (hd_third : r.d / 3 > 0) 
  (hc_positive : r.c > 0) 
  (hd_positive : r.d > 0) :
  r.d / r.c = 2 := by
  sorry

end canister_capacity_ratio_l1556_155683


namespace daps_dops_dips_equivalence_l1556_155637

/-- Given that 5 daps are equivalent to 4 dops and 3 dops are equivalent to 11 dips,
    prove that 22.5 daps are equivalent to 66 dips. -/
theorem daps_dops_dips_equivalence 
  (h1 : (5 : ℚ) / 4 = daps_per_dop) 
  (h2 : (3 : ℚ) / 11 = dops_per_dip) : 
  (66 : ℚ) * daps_per_dop * dops_per_dip = (45 : ℚ) / 2 := by
  sorry

end daps_dops_dips_equivalence_l1556_155637


namespace power_sum_difference_l1556_155631

theorem power_sum_difference : 2^4 + 2^4 + 2^4 - 2^2 = 44 := by
  sorry

end power_sum_difference_l1556_155631


namespace binomial_expansion_ratio_l1556_155645

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -122 / 121 := by
sorry

end binomial_expansion_ratio_l1556_155645


namespace eva_marks_ratio_l1556_155643

/-- Represents the marks Eva scored in a subject for a semester -/
structure Marks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for both semesters -/
structure YearlyMarks where
  first_semester : Marks
  second_semester : Marks

def total_marks (ym : YearlyMarks) : ℕ :=
  ym.first_semester.maths + ym.first_semester.arts + ym.first_semester.science +
  ym.second_semester.maths + ym.second_semester.arts + ym.second_semester.science

theorem eva_marks_ratio :
  ∀ (ym : YearlyMarks),
    ym.second_semester.maths = 80 →
    ym.second_semester.arts = 90 →
    ym.second_semester.science = 90 →
    ym.first_semester.maths = ym.second_semester.maths + 10 →
    ym.first_semester.arts = ym.second_semester.arts - 15 →
    ym.first_semester.science < ym.second_semester.science →
    total_marks ym = 485 →
    ∃ (x : ℕ), 
      ym.second_semester.science - ym.first_semester.science = x ∧
      x = 30 ∧
      (x : ℚ) / ym.second_semester.science = 1 / 3 :=
by sorry


end eva_marks_ratio_l1556_155643


namespace A_intersect_B_is_empty_l1556_155618

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Theorem statement
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by sorry

end A_intersect_B_is_empty_l1556_155618


namespace root_sum_theorem_l1556_155687

theorem root_sum_theorem (m n : ℝ) : 
  (∀ x, x^2 - (m+n)*x + m*n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3*n :=
sorry

end root_sum_theorem_l1556_155687


namespace arithmetic_progression_with_prime_condition_l1556_155676

theorem arithmetic_progression_with_prime_condition :
  ∀ (a b c d : ℤ),
  (∃ (k : ℤ), b = a + k ∧ c = b + k ∧ d = c + k) →  -- arithmetic progression
  (∃ (p : ℕ), Nat.Prime p ∧ (d - c + 1 : ℤ) = p) →  -- d - c + 1 is prime
  a + b^2 + c^3 = d^2 * b →                        -- given equation
  (∃ (n : ℤ), a = n ∧ b = n + 1 ∧ c = n + 2 ∧ d = n + 3) :=
by sorry

end arithmetic_progression_with_prime_condition_l1556_155676


namespace perpendicular_vectors_l1556_155615

/-- Given two vectors a and b in R², prove that if k*a + b is perpendicular to a - b, then k = 1. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 1))
  (h2 : b = (-1, 1))
  (h3 : (k * a.1 + b.1) * (a.1 - b.1) + (k * a.2 + b.2) * (a.2 - b.2) = 0) :
  k = 1 := by
  sorry

end perpendicular_vectors_l1556_155615


namespace pascal_triangle_51st_row_third_number_l1556_155673

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 51
  let k : ℕ := 2
  Nat.choose n k = 1275 := by sorry

end pascal_triangle_51st_row_third_number_l1556_155673


namespace xy_less_than_one_necessary_not_sufficient_l1556_155651

theorem xy_less_than_one_necessary_not_sufficient (x y : ℝ) :
  (0 < x ∧ x < 1/y) → (x*y < 1) ∧
  ¬(∀ x y : ℝ, x*y < 1 → (0 < x ∧ x < 1/y)) :=
by sorry

end xy_less_than_one_necessary_not_sufficient_l1556_155651


namespace quagga_placements_l1556_155693

/-- Represents a chessboard --/
def Chessboard := Fin 8 × Fin 8

/-- Represents a quagga's move --/
def QuaggaMove := (Int × Int) × (Int × Int)

/-- Defines the valid moves for a quagga --/
def validQuaggaMoves : List QuaggaMove :=
  [(( 6,  0), ( 0,  5)), (( 6,  0), ( 0, -5)),
   ((-6,  0), ( 0,  5)), ((-6,  0), ( 0, -5)),
   (( 0,  6), ( 5,  0)), (( 0,  6), (-5,  0)),
   (( 0, -6), ( 5,  0)), (( 0, -6), (-5,  0))]

/-- Checks if a move is valid on the chessboard --/
def isValidMove (start : Chessboard) (move : QuaggaMove) : Bool :=
  let ((dx1, dy1), (dx2, dy2)) := move
  let (x, y) := start
  let x1 := x + dx1
  let y1 := y + dy1
  let x2 := x1 + dx2
  let y2 := y1 + dy2
  0 ≤ x2 ∧ x2 < 8 ∧ 0 ≤ y2 ∧ y2 < 8

/-- Represents a placement of quaggas on the chessboard --/
def QuaggaPlacement := List Chessboard

/-- Checks if a placement is valid (no quaggas attack each other) --/
def isValidPlacement (placement : QuaggaPlacement) : Bool :=
  sorry

/-- The main theorem to prove --/
theorem quagga_placements :
  (∃ (placements : List QuaggaPlacement),
    placements.length = 68 ∧
    ∀ p ∈ placements,
      p.length = 51 ∧
      isValidPlacement p) :=
sorry

end quagga_placements_l1556_155693


namespace cover_ways_eq_fib_succ_l1556_155682

/-- The number of ways to cover a 2 × n grid with 1 × 2 tiles -/
def cover_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => cover_ways (n+1) + cover_ways n

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

theorem cover_ways_eq_fib_succ (n : ℕ) : cover_ways n = fib (n+1) := by
  sorry

#eval cover_ways 10  -- Should evaluate to 89

end cover_ways_eq_fib_succ_l1556_155682


namespace ernies_income_ratio_l1556_155671

def ernies_previous_income : ℕ := 6000
def jacks_income : ℕ := 2 * ernies_previous_income
def combined_income : ℕ := 16800
def ernies_current_income : ℕ := combined_income - jacks_income

theorem ernies_income_ratio :
  (ernies_current_income : ℚ) / ernies_previous_income = 2 / 3 := by sorry

end ernies_income_ratio_l1556_155671


namespace emani_money_l1556_155649

/-- Proves that Emani has $150, given the conditions of the problem -/
theorem emani_money :
  (∀ (emani howard : ℕ),
    emani = howard + 30 →
    emani + howard = 2 * 135 →
    emani = 150) :=
by sorry

end emani_money_l1556_155649


namespace tangent_sum_simplification_l1556_155648

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (50 * π / 180) + 
   Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / 
  Real.sin (30 * π / 180) = 
  2 * (1 / Real.cos (50 * π / 180) + 
       1 / (2 * Real.cos (70 * π / 180) * Real.cos (80 * π / 180))) := by
  sorry

end tangent_sum_simplification_l1556_155648


namespace factor_condition_l1556_155610

/-- A quadratic trinomial can be factored using the cross multiplication method if
    there exist two integers that multiply to give the constant term and add up to
    the coefficient of x. -/
def can_be_factored_by_cross_multiplication (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), p * q = c ∧ p + q = b

/-- If x^2 + kx + 5 can be factored using the cross multiplication method,
    then k = 6 or k = -6 -/
theorem factor_condition (k : ℤ) :
  can_be_factored_by_cross_multiplication 1 k 5 → k = 6 ∨ k = -6 := by
  sorry

end factor_condition_l1556_155610


namespace k_value_for_decreasing_function_l1556_155667

theorem k_value_for_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)
  (h_domain : ∀ x, x ≤ 1 → ∃ y, f x = y)
  (h_inequality : ∀ x : ℝ, f (k - Real.sin x) ≥ f (k^2 - Real.sin x^2))
  : k = -1 :=
sorry

end k_value_for_decreasing_function_l1556_155667


namespace range_of_a_l1556_155622

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l1556_155622


namespace right_rectangular_prism_volume_l1556_155613

theorem right_rectangular_prism_volume (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 54 →
  b * c = 56 →
  a * c = 60 →
  abs (a * b * c - 426) < 0.5 :=
by sorry

end right_rectangular_prism_volume_l1556_155613


namespace x_minus_25_is_perfect_square_l1556_155661

/-- Represents the number of zeros after the first 1 in the definition of x -/
def zeros_after_first_one : ℕ := 2011

/-- Represents the number of zeros after the second 1 in the definition of x -/
def zeros_after_second_one : ℕ := 2012

/-- Defines x as described in the problem -/
def x : ℕ := 
  10^(zeros_after_second_one + 3) + 
  10^(zeros_after_first_one + zeros_after_second_one + 2) + 
  50

/-- States that x - 25 is a perfect square -/
theorem x_minus_25_is_perfect_square : 
  ∃ n : ℕ, x - 25 = n^2 := by sorry

end x_minus_25_is_perfect_square_l1556_155661


namespace complement_union_theorem_l1556_155669

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end complement_union_theorem_l1556_155669


namespace tan_difference_pi_12_5pi_12_l1556_155617

theorem tan_difference_pi_12_5pi_12 : 
  Real.tan (π / 12) - Real.tan (5 * π / 12) = -2 * Real.sqrt 3 := by
  sorry

end tan_difference_pi_12_5pi_12_l1556_155617


namespace consecutive_integers_product_210_l1556_155689

/-- Given three consecutive integers whose product is 210 and whose sum of squares is minimized,
    the sum of the two smallest of these integers is 11. -/
theorem consecutive_integers_product_210 (n : ℤ) :
  (n - 1) * n * (n + 1) = 210 ∧
  ∀ m : ℤ, (m - 1) * m * (m + 1) = 210 → 
    (n - 1)^2 + n^2 + (n + 1)^2 ≤ (m - 1)^2 + m^2 + (m + 1)^2 →
  (n - 1) + n = 11 := by
  sorry

end consecutive_integers_product_210_l1556_155689
