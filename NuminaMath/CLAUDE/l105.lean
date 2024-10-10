import Mathlib

namespace heidi_painting_fraction_l105_10545

/-- Represents the time in minutes it takes Heidi to paint a wall -/
def total_time : ℚ := 45

/-- Represents the time in minutes we want to calculate the painted fraction for -/
def given_time : ℚ := 9

/-- Represents the fraction of the wall painted in the given time -/
def painted_fraction : ℚ := given_time / total_time

theorem heidi_painting_fraction :
  painted_fraction = 1 / 5 := by sorry

end heidi_painting_fraction_l105_10545


namespace hostel_provisions_l105_10523

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 28

/-- The number of days the provisions would last if 50 men left -/
def extended_days : ℕ := 35

/-- The number of men that would leave -/
def men_leaving : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_leaving) * extended_days := by
  sorry

end hostel_provisions_l105_10523


namespace ellipse_equation_proof_l105_10537

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 81 + y^2 / 45 = 1

/-- Given ellipse satisfies the conditions -/
def given_ellipse : Ellipse :=
  { center := (0, 0)
  , foci := ((-3, 0), (3, 0))
  , point := (3, 8) }

/-- Theorem: The equation of the given ellipse is x²/81 + y²/45 = 1 -/
theorem ellipse_equation_proof :
  ellipse_equation given_ellipse (given_ellipse.point.1) (given_ellipse.point.2) :=
by sorry

end ellipse_equation_proof_l105_10537


namespace infinitely_many_numbers_with_property_l105_10544

/-- A function that returns the number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of prime factors of a natural number -/
def prodPrimeFactors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of exponents in the prime factorization of a natural number -/
def prodExponents (n : ℕ) : ℕ := sorry

/-- The property that we want to prove holds for infinitely many natural numbers -/
def hasProperty (n : ℕ) : Prop :=
  numDivisors n = prodPrimeFactors n - prodExponents n

theorem infinitely_many_numbers_with_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, hasProperty n := by sorry

end infinitely_many_numbers_with_property_l105_10544


namespace quadratic_inequality_range_l105_10524

theorem quadratic_inequality_range (θ : Real) :
  (∀ m : Real, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (∀ m : Real, m ≥ 4 ∨ m ≤ 0) :=
by sorry

end quadratic_inequality_range_l105_10524


namespace second_month_sale_l105_10538

def average_sale : ℕ := 5900
def first_month : ℕ := 5921
def third_month : ℕ := 5568
def fourth_month : ℕ := 6088
def fifth_month : ℕ := 6433
def sixth_month : ℕ := 5922

theorem second_month_sale :
  ∃ (second_month : ℕ),
    second_month = 
      6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month) ∧
    second_month = 5468 := by
  sorry

end second_month_sale_l105_10538


namespace intersection_of_M_and_N_l105_10579

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - 1)}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/2 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l105_10579


namespace employee_salary_proof_l105_10585

/-- The weekly salary of employee n -/
def salary_n : ℝ := 270

/-- The weekly salary of employee m -/
def salary_m : ℝ := 1.2 * salary_n

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 594

theorem employee_salary_proof :
  salary_n + salary_m = total_salary :=
by sorry

end employee_salary_proof_l105_10585


namespace only_set_B_forms_triangle_l105_10575

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def set_A : List ℝ := [2, 6, 8]
def set_B : List ℝ := [4, 6, 7]
def set_C : List ℝ := [5, 6, 12]
def set_D : List ℝ := [2, 3, 6]

theorem only_set_B_forms_triangle :
  (¬ triangle_inequality set_A[0] set_A[1] set_A[2]) ∧
  (triangle_inequality set_B[0] set_B[1] set_B[2]) ∧
  (¬ triangle_inequality set_C[0] set_C[1] set_C[2]) ∧
  (¬ triangle_inequality set_D[0] set_D[1] set_D[2]) :=
by sorry

end only_set_B_forms_triangle_l105_10575


namespace tadd_250th_number_l105_10569

/-- Represents the block size for a player in the n-th round -/
def blockSize (n : ℕ) : ℕ := 6 * n - 5

/-- Sum of numbers spoken up to the k-th block -/
def sumUpToBlock (k : ℕ) : ℕ := 3 * k * (k - 1)

/-- The counting game as described in the problem -/
def countingGame : Prop :=
  ∃ (k : ℕ),
    sumUpToBlock (k - 1) < 250 ∧
    250 ≤ sumUpToBlock k ∧
    250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))

theorem tadd_250th_number :
  countingGame → (∃ (k : ℕ), 250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))) :=
by sorry

end tadd_250th_number_l105_10569


namespace fraction_evaluation_l105_10506

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l105_10506


namespace lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l105_10587

theorem lcm_gcd_fraction_lower_bound (a b c : ℕ+) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) ≥ 5 / 2 :=
sorry

theorem lcm_gcd_fraction_bound_achievable :
  ∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = 5 / 2 :=
sorry

end lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l105_10587


namespace angelinas_journey_l105_10551

/-- Angelina's journey with varying speeds -/
theorem angelinas_journey (distance_home_grocery : ℝ) (distance_grocery_gym : ℝ) (time_difference : ℝ) :
  distance_home_grocery = 840 →
  distance_grocery_gym = 480 →
  time_difference = 40 →
  ∃ (speed_home_grocery : ℝ),
    speed_home_grocery > 0 ∧
    distance_home_grocery / speed_home_grocery - distance_grocery_gym / (2 * speed_home_grocery) = time_difference ∧
    2 * speed_home_grocery = 30 :=
by sorry

end angelinas_journey_l105_10551


namespace kiwi_profit_optimization_l105_10539

/-- Kiwi prices and profit optimization problem -/
theorem kiwi_profit_optimization 
  (green_price red_price : ℕ) 
  (green_cost red_cost : ℕ) 
  (total_boxes : ℕ) 
  (max_expenditure : ℕ) :
  green_cost = 80 →
  red_cost = 100 →
  total_boxes = 21 →
  max_expenditure = 2000 →
  red_price = green_price + 25 →
  6 * green_price = 5 * red_price - 25 →
  green_price = 100 ∧ 
  red_price = 125 ∧
  (∃ (green_boxes red_boxes : ℕ),
    green_boxes + red_boxes = total_boxes ∧
    green_boxes * green_cost + red_boxes * red_cost ≤ max_expenditure ∧
    green_boxes = 5 ∧ 
    red_boxes = 16 ∧
    (green_boxes * (green_price - green_cost) + red_boxes * (red_price - red_cost)) = 500 ∧
    ∀ (g r : ℕ), 
      g + r = total_boxes → 
      g * green_cost + r * red_cost ≤ max_expenditure →
      g * (green_price - green_cost) + r * (red_price - red_cost) ≤ 500) :=
by sorry

end kiwi_profit_optimization_l105_10539


namespace delta_quotient_equals_two_plus_delta_x_l105_10529

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, given points (1, 2) and (1 + Δx, 2 + Δy) on the graph,
    Δy / Δx = 2 + Δx for any non-zero Δx -/
theorem delta_quotient_equals_two_plus_delta_x (Δx : ℝ) (Δy : ℝ) (h : Δx ≠ 0) :
  f (1 + Δx) = 2 + Δy →
  Δy / Δx = 2 + Δx :=
by sorry

end delta_quotient_equals_two_plus_delta_x_l105_10529


namespace probability_one_ball_in_last_box_l105_10580

/-- The probability of exactly one ball landing in the last box when 100 balls
    are randomly distributed among 100 boxes. -/
theorem probability_one_ball_in_last_box :
  let n : ℕ := 100
  let p : ℝ := 1 / n
  (n : ℝ) * p * (1 - p) ^ (n - 1) = (1 - 1 / n) ^ (n - 1) := by sorry

end probability_one_ball_in_last_box_l105_10580


namespace block_final_height_l105_10512

/-- Given a block sliding down one ramp and up another, both at angle θ,
    with initial height h₁, mass m, and coefficient of kinetic friction μₖ,
    the final height h₂ is given by h₂ = h₁ / (1 + μₖ * √3) -/
theorem block_final_height
  (m : ℝ) (h₁ : ℝ) (μₖ : ℝ) (θ : ℝ) 
  (h₁_pos : h₁ > 0)
  (m_pos : m > 0)
  (μₖ_pos : μₖ > 0)
  (θ_val : θ = π/6) :
  let h₂ := h₁ / (1 + μₖ * Real.sqrt 3)
  ∀ ε > 0, abs (h₂ - h₁ / (1 + μₖ * Real.sqrt 3)) < ε :=
by
  sorry

#check block_final_height

end block_final_height_l105_10512


namespace sum_of_cubes_negative_l105_10541

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
sorry

end sum_of_cubes_negative_l105_10541


namespace bottle_cap_distribution_l105_10521

theorem bottle_cap_distribution (total_caps : ℕ) (total_boxes : ℕ) (caps_per_box : ℕ) : 
  total_caps = 60 → 
  total_boxes = 60 → 
  total_caps = total_boxes * caps_per_box → 
  caps_per_box = 1 := by
  sorry

end bottle_cap_distribution_l105_10521


namespace polynomial_identity_sum_of_squares_l105_10516

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) → 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
sorry

end polynomial_identity_sum_of_squares_l105_10516


namespace total_annual_interest_l105_10505

theorem total_annual_interest (total_amount first_part : ℕ) : 
  total_amount = 4000 →
  first_part = 2800 →
  (first_part * 3 + (total_amount - first_part) * 5) / 100 = 144 := by
sorry

end total_annual_interest_l105_10505


namespace last_page_cards_l105_10507

/-- Represents the number of cards that can be placed on different page types -/
inductive PageType
| Four : PageType
| Six : PageType
| Eight : PageType

/-- Calculates the number of cards on the last partially-filled page -/
def cardsOnLastPage (totalCards : ℕ) (pageTypes : List PageType) : ℕ :=
  sorry

/-- Theorem stating that for 137 cards and the given page types, 
    the number of cards on the last partially-filled page is 1 -/
theorem last_page_cards : 
  cardsOnLastPage 137 [PageType.Four, PageType.Six, PageType.Eight] = 1 := by
  sorry

end last_page_cards_l105_10507


namespace vowel_initial_probability_l105_10501

/-- The probability of selecting a student with vowel initials -/
theorem vowel_initial_probability 
  (total_students : ℕ) 
  (vowels : List Char) 
  (students_per_vowel : ℕ) : 
  total_students = 34 → 
  vowels = ['A', 'E', 'I', 'O', 'U', 'Y'] → 
  students_per_vowel = 2 → 
  (students_per_vowel * vowels.length : ℚ) / total_students = 6 / 17 := by
  sorry

end vowel_initial_probability_l105_10501


namespace max_value_product_l105_10534

theorem max_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 2) : 
  x^4 * y^3 * z^2 ≤ (1 : ℝ) / 9765625 :=
sorry

end max_value_product_l105_10534


namespace sum_of_squares_inequality_range_of_a_l105_10571

-- Part I
theorem sum_of_squares_inequality (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end sum_of_squares_inequality_range_of_a_l105_10571


namespace lottery_probability_l105_10553

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_draw : ℕ := 6
  1 / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 419514480 :=
by sorry

end lottery_probability_l105_10553


namespace rectangle_area_l105_10508

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 1296
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_area : ℝ := rectangle_length * b
  rectangle_area = 6 * b :=
by sorry

end rectangle_area_l105_10508


namespace total_pages_left_to_read_l105_10502

/-- Calculates the total number of pages left to read from three books -/
def pagesLeftToRead (book1Total book1Read book2Total book2Read book3Total book3Read : ℕ) : ℕ :=
  (book1Total - book1Read) + (book2Total - book2Read) + (book3Total - book3Read)

/-- Theorem: The total number of pages left to read from three books is 1442 -/
theorem total_pages_left_to_read : 
  pagesLeftToRead 563 147 849 389 700 134 = 1442 := by
  sorry

end total_pages_left_to_read_l105_10502


namespace elena_max_flour_l105_10520

/-- Represents the recipe and available ingredients for Elena's bread --/
structure BreadRecipe where
  butter_ratio : ℚ  -- Ratio of butter to flour (in ounces per cup)
  sugar_ratio : ℚ   -- Ratio of sugar to flour (in ounces per cup)
  available_butter : ℚ  -- Available butter in ounces
  available_sugar : ℚ   -- Available sugar in ounces

/-- Calculates the maximum cups of flour that can be used given the recipe and available ingredients --/
def max_flour (recipe : BreadRecipe) : ℚ :=
  min 
    (recipe.available_butter / recipe.butter_ratio)
    (recipe.available_sugar / recipe.sugar_ratio)

/-- Elena's specific bread recipe and available ingredients --/
def elena_recipe : BreadRecipe :=
  { butter_ratio := 3/4
  , sugar_ratio := 2/5
  , available_butter := 24
  , available_sugar := 30 }

/-- Theorem stating that the maximum number of cups of flour Elena can use is 32 --/
theorem elena_max_flour : 
  max_flour elena_recipe = 32 := by sorry

end elena_max_flour_l105_10520


namespace unique_solution_for_system_l105_10509

theorem unique_solution_for_system : ∃! y : ℚ, 9 * y^2 + 8 * y - 2 = 0 ∧ 27 * y^2 + 62 * y - 8 = 0 :=
  by sorry

end unique_solution_for_system_l105_10509


namespace f_properties_l105_10510

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∃ (t : ℝ), t > 0 ∧ ∀ (x : ℝ), f (x + t) = f x ∧ ∀ (s : ℝ), s > 0 ∧ (∀ (x : ℝ), f (x + s) = f x) → t ≤ s) ∧
  (∀ (x : ℝ), x ≥ -π/12 ∧ x ≤ 5*π/12 → f x ≥ -1/2 ∧ f x ≤ 1) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≥ -π/12 ∧ x₁ ≤ 5*π/12 ∧ x₂ ≥ -π/12 ∧ x₂ ≤ 5*π/12 ∧ f x₁ = -1/2 ∧ f x₂ = 1) :=
by sorry


end f_properties_l105_10510


namespace maisie_flyers_l105_10578

theorem maisie_flyers : 
  ∀ (maisie_flyers : ℕ), 
  (71 : ℕ) = 2 * maisie_flyers + 5 → 
  maisie_flyers = 33 :=
by
  sorry

end maisie_flyers_l105_10578


namespace survivor_quitters_probability_l105_10517

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (total_people : ℕ) (tribe_size : ℕ) (quitters : ℕ)
  (h1 : total_people = 20)
  (h2 : tribe_size = 10)
  (h3 : quitters = 3)
  (h4 : total_people = 2 * tribe_size) :
  (Nat.choose tribe_size quitters * 2 : ℚ) / Nat.choose total_people quitters = 20 / 95 := by
  sorry

end survivor_quitters_probability_l105_10517


namespace sequence_sum_l105_10592

/-- Given a sequence {a_n} with sum of first n terms S_n, prove S_n = -3^(n-1) -/
theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 1 = -1) → 
  (∀ n : ℕ, a (n + 1) = 2 * S n) → 
  (∀ n : ℕ, S n = -(3^(n - 1))) := by
  sorry

end sequence_sum_l105_10592


namespace f_of_three_eq_seventeen_l105_10564

/-- Given a function f(x) = ax + bx + c where c is a constant,
    if f(1) = 7 and f(2) = 12, then f(3) = 17 -/
theorem f_of_three_eq_seventeen
  (f : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, f x = a * x + b * x + c)
  (h2 : f 1 = 7)
  (h3 : f 2 = 12) :
  f 3 = 17 := by
sorry

end f_of_three_eq_seventeen_l105_10564


namespace geometric_series_ratio_l105_10591

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (hr2 : r ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end geometric_series_ratio_l105_10591


namespace union_complement_equal_l105_10559

def U : Finset Nat := {0,1,2,4,6,8}
def M : Finset Nat := {0,4,6}
def N : Finset Nat := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end union_complement_equal_l105_10559


namespace triangle_area_in_nested_rectangles_l105_10560

/-- Given a rectangle with dimensions a × b and a smaller rectangle inside with dimensions u × v,
    where the sides are parallel, the area of one of the four congruent right triangles formed by
    connecting the vertices of the smaller rectangle to the midpoints of the sides of the larger
    rectangle is (a-u)(b-v)/8. -/
theorem triangle_area_in_nested_rectangles (a b u v : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hu : 0 < u) (hv : 0 < v) (hu_lt_a : u < a) (hv_lt_b : v < b) :
  (a - u) * (b - v) / 8 = (a - u) * (b - v) / 8 := by sorry

end triangle_area_in_nested_rectangles_l105_10560


namespace union_equality_implies_a_values_l105_10574

theorem union_equality_implies_a_values (a : ℝ) : 
  ({1, a} : Set ℝ) ∪ {a^2} = {1, a} → a = -1 ∨ a = 0 :=
by sorry

end union_equality_implies_a_values_l105_10574


namespace min_value_absolute_sum_l105_10565

theorem min_value_absolute_sum (x : ℝ) :
  |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ≥ 45 / 8 ∧
  ∃ x : ℝ, |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| = 45 / 8 :=
by sorry

end min_value_absolute_sum_l105_10565


namespace X_equals_three_l105_10554

/-- The length of the unknown segment X in the diagram --/
def X : ℝ := sorry

/-- The total length of the top side of the figure --/
def top_length : ℝ := 3 + 2 + X + 4

/-- The length of the bottom side of the figure --/
def bottom_length : ℝ := 12

/-- Theorem stating that X equals 3 --/
theorem X_equals_three : X = 3 := by
  sorry

end X_equals_three_l105_10554


namespace trig_expression_equals_one_l105_10586

theorem trig_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end trig_expression_equals_one_l105_10586


namespace initial_bananas_count_l105_10536

/-- The number of bananas Elizabeth bought initially -/
def initial_bananas : ℕ := sorry

/-- The number of bananas Elizabeth ate -/
def eaten_bananas : ℕ := 4

/-- The number of bananas Elizabeth has left -/
def remaining_bananas : ℕ := 8

/-- Theorem stating that the initial number of bananas is 12 -/
theorem initial_bananas_count : initial_bananas = 12 := by sorry

end initial_bananas_count_l105_10536


namespace line_slope_intercept_product_l105_10549

/-- Given a line with equation y = mx + b, where m = -2/3 and b = 3/2, prove that mb = -1 -/
theorem line_slope_intercept_product (m b : ℚ) : 
  m = -2/3 → b = 3/2 → m * b = -1 := by
  sorry

end line_slope_intercept_product_l105_10549


namespace triangle_circle_relation_l105_10519

theorem triangle_circle_relation (α β γ s R r : ℝ) :
  -- Triangle angles
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π →
  -- Perimeter is 2s
  s > 0 →
  -- R is the radius of the circumscribed circle
  R > 0 →
  -- r is the radius of the inscribed circle
  r > 0 →
  -- The theorem
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = s^2 - (r + 2*R)^2 := by
  sorry

end triangle_circle_relation_l105_10519


namespace traffic_light_probability_l105_10584

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_times : List ℕ

/-- Calculates the probability of observing a color change in a given interval -/
def probability_of_change (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  let change_windows := cycle.change_times.map (λ t => if t ≤ cycle.total_time - interval then interval else t + interval - cycle.total_time)
  let total_change_time := change_windows.sum
  total_change_time / cycle.total_time

/-- The main theorem: probability of observing a color change is 1/7 -/
theorem traffic_light_probability :
  let cycle : TrafficLightCycle := { total_time := 63, change_times := [30, 33, 63] }
  probability_of_change cycle 3 = 1/7 := by
  sorry


end traffic_light_probability_l105_10584


namespace flag_arrangement_count_l105_10532

def total_arrangements (n m : ℕ) : ℕ := (n + m).factorial / (n.factorial * m.factorial)

def consecutive_arrangements (n m : ℕ) : ℕ := (m + 1).factorial / m.factorial

theorem flag_arrangement_count : 
  let total := total_arrangements 3 4
  let red_consecutive := consecutive_arrangements 1 4
  let blue_consecutive := consecutive_arrangements 3 1
  let both_consecutive := 2
  total - red_consecutive - blue_consecutive + both_consecutive = 28 := by sorry

end flag_arrangement_count_l105_10532


namespace part1_part2_l105_10558

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 3| - |2*x - a|

-- Part 1
theorem part1 (a : ℝ) : 
  (∃ x, f a x ≤ -5) → (a ≤ -8 ∨ a ≥ 2) := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, f a (x - 1/2) + f a (-x - 1/2) = 0) → a = 1 := by sorry

end part1_part2_l105_10558


namespace fish_problem_l105_10576

/-- The number of fish originally in the shop -/
def original_fish : ℕ := 36

/-- The number of fish remaining after lunch sale -/
def after_lunch (f : ℕ) : ℕ := f / 2

/-- The number of fish sold for dinner -/
def dinner_sale (f : ℕ) : ℕ := (after_lunch f) / 3

/-- The number of fish remaining after both sales -/
def remaining_fish (f : ℕ) : ℕ := (after_lunch f) - (dinner_sale f)

theorem fish_problem :
  remaining_fish original_fish = 12 :=
by sorry

end fish_problem_l105_10576


namespace magnitude_of_complex_reciprocal_l105_10590

open Complex

theorem magnitude_of_complex_reciprocal (z : ℂ) : z = (1 : ℂ) / (1 - I) → abs z = Real.sqrt 2 / 2 := by
  sorry

end magnitude_of_complex_reciprocal_l105_10590


namespace investment_comparison_l105_10550

/-- Represents the final value of an investment after two years --/
def final_value (initial : ℝ) (change1 : ℝ) (change2 : ℝ) (dividend_rate : ℝ) : ℝ :=
  let value1 := initial * (1 + change1)
  let value2 := value1 * (1 + change2)
  value2 + value1 * dividend_rate

/-- Theorem stating the relationship between final investment values --/
theorem investment_comparison : 
  let a := final_value 200 0.15 (-0.10) 0.05
  let b := final_value 150 (-0.20) 0.30 0
  let c := final_value 100 0 0 0
  c < b ∧ b < a := by sorry

end investment_comparison_l105_10550


namespace expression_equals_one_l105_10515

theorem expression_equals_one :
  (121^2 - 11^2) / (91^2 - 13^2) * ((91-13)*(91+13)) / ((121-11)*(121+11)) = 1 := by
  sorry

end expression_equals_one_l105_10515


namespace trigonometric_identities_l105_10595

noncomputable def α : ℝ := Real.arcsin (2/3 - Real.sqrt (5/9))
noncomputable def β : ℝ := Real.arctan 2

theorem trigonometric_identities 
  (h1 : Real.sin α + Real.cos α = 2/3)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.tan β = 2) :
  (Real.sin (3*π/2 - α) * Real.cos (-π/2 - α) = -5/18) ∧
  ((1 / Real.sin (π - α)) - (1 / Real.cos (2*π - α)) + 
   (Real.sin β - Real.cos β) / (2*Real.sin β + Real.cos β) = (6*Real.sqrt 14 + 1)/5) := by
  sorry

end trigonometric_identities_l105_10595


namespace right_triangle_area_l105_10573

theorem right_triangle_area (leg : ℝ) (altitude : ℝ) (area : ℝ) : 
  leg = 15 → altitude = 9 → area = 84.375 → 
  ∃ (hypotenuse : ℝ), 
    hypotenuse * altitude / 2 = area ∧ 
    leg^2 + altitude^2 = (hypotenuse / 2)^2 := by
  sorry

end right_triangle_area_l105_10573


namespace transformed_roots_l105_10583

-- Define the original quadratic equation
def original_quadratic (p q r x : ℝ) : Prop := p * x^2 + q * x + r = 0

-- Define the roots of the original quadratic equation
def has_roots (p q r u v : ℝ) : Prop := original_quadratic p q r u ∧ original_quadratic p q r v

-- Define the new quadratic equation
def new_quadratic (p q r x : ℝ) : Prop := x^2 - 4 * q * x + 4 * p * r + 3 * q^2 = 0

-- Theorem statement
theorem transformed_roots (p q r u v : ℝ) (hp : p ≠ 0) :
  has_roots p q r u v →
  new_quadratic p q r (2 * p * u + 3 * q) ∧ new_quadratic p q r (2 * p * v + 3 * q) :=
by sorry

end transformed_roots_l105_10583


namespace binary_101111_is_47_l105_10504

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111_is_47 :
  binary_to_decimal [true, true, true, true, true, false] = 47 := by
  sorry

end binary_101111_is_47_l105_10504


namespace max_sum_product_l105_10599

theorem max_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a ≥ 50 → 
  a * b + b * c + c * d + d * a ≤ 5000 := by
sorry

end max_sum_product_l105_10599


namespace mans_speed_against_current_l105_10525

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: The man's speed against the current is 20 km/hr given the conditions -/
theorem mans_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end mans_speed_against_current_l105_10525


namespace tangent_perpendicular_implies_a_value_l105_10557

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

-- Define the derivative of f(x)
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (k^2 - 1) * x

-- Theorem statement
theorem tangent_perpendicular_implies_a_value (k : ℝ) (a : ℝ) (b : ℝ) :
  f k 1 = a →                   -- Point (1, a) is on the graph of f
  f_deriv k 1 = -1 →            -- Tangent line is perpendicular to x - y + b = 0
  a = -2 := by
sorry

end tangent_perpendicular_implies_a_value_l105_10557


namespace absolute_value_of_3_plus_i_l105_10572

theorem absolute_value_of_3_plus_i :
  let z : ℂ := 3 + Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end absolute_value_of_3_plus_i_l105_10572


namespace intersection_point_coordinates_l105_10598

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci of the hyperbola
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define point A
def point_A : ℝ × ℝ := (6, -2)

-- Define the line passing through right focus and point A
def line_through_right_focus (x y : ℝ) : Prop := 2*x + y - 10 = 0

-- Define the perpendicular line passing through left focus
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- The theorem to prove
theorem intersection_point_coordinates :
  ∃ (x y : ℝ), 
    hyperbola x y ∧
    line_through_right_focus x y ∧
    perpendicular_line x y ∧
    x = 3 ∧ y = 4 := by sorry

end intersection_point_coordinates_l105_10598


namespace binomial_square_constant_l105_10596

theorem binomial_square_constant (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + k = (a*x + b)^2) → k = 900 := by
  sorry

end binomial_square_constant_l105_10596


namespace min_blocks_for_slotted_structure_l105_10535

/-- A block with one hook and five slots -/
structure Block :=
  (hook : Fin 6)
  (slots : Finset (Fin 6))
  (hook_slot_distinct : hook ∉ slots)
  (slot_count : slots.card = 5)

/-- A structure made of blocks -/
structure Structure :=
  (blocks : Finset Block)
  (no_visible_hooks : ∀ b ∈ blocks, ∃ b' ∈ blocks, b.hook ∈ b'.slots)

/-- The theorem stating that the minimum number of blocks required is 4 -/
theorem min_blocks_for_slotted_structure :
  ∀ s : Structure, s.blocks.card ≥ 4 ∧ 
  ∃ s' : Structure, s'.blocks.card = 4 :=
sorry

end min_blocks_for_slotted_structure_l105_10535


namespace quadratic_translation_l105_10568

-- Define the original quadratic function
def f (x : ℝ) : ℝ := (x - 2009) * (x - 2008) + 4

-- Define the translated function
def g (x : ℝ) : ℝ := f x - 4

-- Theorem statement
theorem quadratic_translation :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ |x₁ - x₂| = 1) :=
sorry

end quadratic_translation_l105_10568


namespace quadratic_inequality_solution_l105_10540

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | f a x > 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  solution_set a = 
    if -1 < a then {x | 1 < x ∧ x < -1/a}
    else if a = -1 then ∅
    else {x | -1/a < x ∧ x < 1} :=
by sorry

end quadratic_inequality_solution_l105_10540


namespace wind_velocity_theorem_l105_10511

-- Define the relationship between pressure, area, and velocity
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

-- Given initial condition
def initial_condition (k : ℝ) : Prop :=
  pressure_relation k 9 105 = 4

-- Theorem to prove
theorem wind_velocity_theorem (k : ℝ) (h : initial_condition k) :
  pressure_relation k 36 70 = 64 := by
  sorry

end wind_velocity_theorem_l105_10511


namespace fraction_equation_l105_10577

theorem fraction_equation : 45 / (8 - 3/7) = 315/53 := by sorry

end fraction_equation_l105_10577


namespace min_box_height_is_19_l105_10563

/-- Represents the specifications for packaging a fine arts collection --/
structure PackagingSpecs where
  totalVolume : ℝ  -- Total volume needed in cubic inches
  boxCost : ℝ      -- Cost per box in dollars
  totalCost : ℝ    -- Total cost spent on boxes in dollars

/-- Calculates the minimum height of cubic boxes needed to package a collection --/
def minBoxHeight (specs : PackagingSpecs) : ℕ :=
  sorry

/-- Theorem stating that the minimum box height for the given specifications is 19 inches --/
theorem min_box_height_is_19 :
  let specs : PackagingSpecs := {
    totalVolume := 3060000,  -- 3.06 million cubic inches
    boxCost := 0.5,          -- $0.50 per box
    totalCost := 255         -- $255 total cost
  }
  minBoxHeight specs = 19 := by sorry

end min_box_height_is_19_l105_10563


namespace function_max_value_l105_10547

-- Define the function f
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

-- State the theorem
theorem function_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 20) →
  m = -2 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x (-2) ≤ 20 :=
by sorry

end function_max_value_l105_10547


namespace factors_of_M_l105_10588

/-- The number of natural-number factors of M, where M = 2^4 * 3^3 * 5^2 * 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * 4 * 3 * 2

/-- M is defined as 2^4 * 3^3 * 5^2 * 7^1 -/
def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem factors_of_M : num_factors M = 120 := by
  sorry

end factors_of_M_l105_10588


namespace percent_equality_l105_10562

theorem percent_equality (x : ℝ) : (60 / 100 * 600 = 50 / 100 * x) → x = 720 := by
  sorry

end percent_equality_l105_10562


namespace pet_store_ratios_l105_10527

/-- Given the ratios of cats to dogs and dogs to parrots, and the number of cats,
    this theorem proves the number of dogs and parrots. -/
theorem pet_store_ratios (cats : ℕ) (dogs : ℕ) (parrots : ℕ) : 
  (3 : ℚ) / 4 = cats / dogs →  -- ratio of cats to dogs
  (2 : ℚ) / 5 = dogs / parrots →  -- ratio of dogs to parrots
  cats = 18 →  -- number of cats
  dogs = 24 ∧ parrots = 60 := by
  sorry

end pet_store_ratios_l105_10527


namespace line_through_point_l105_10522

theorem line_through_point (b : ℚ) : 
  (b * (-3) - (b - 1) * 5 = b - 3) ↔ (b = 8 / 9) := by sorry

end line_through_point_l105_10522


namespace old_lamp_height_l105_10582

theorem old_lamp_height (new_lamp_height : Real) (height_difference : Real) :
  new_lamp_height = 2.33 →
  height_difference = 1.33 →
  new_lamp_height - height_difference = 1.00 := by
  sorry

end old_lamp_height_l105_10582


namespace computer_price_increase_l105_10513

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 520) (h2 : d > 0) : 
  (338 - d) / d * 100 = 30 := by
  sorry

end computer_price_increase_l105_10513


namespace greater_sum_from_inequalities_l105_10503

theorem greater_sum_from_inequalities (a b c d : ℝ) 
  (h1 : a^2 + b > c^2 + d) 
  (h2 : a + b^2 > c + d^2) 
  (h3 : a ≥ (1/2 : ℝ)) 
  (h4 : b ≥ (1/2 : ℝ)) 
  (h5 : c ≥ (1/2 : ℝ)) 
  (h6 : d ≥ (1/2 : ℝ)) : 
  a + b > c + d := by
  sorry

end greater_sum_from_inequalities_l105_10503


namespace min_intersection_points_l105_10556

/-- Represents a configuration of circles on a plane -/
structure CircleConfiguration where
  num_circles : ℕ
  num_intersections : ℕ
  intersections_per_circle : ℕ → ℕ

/-- The minimum number of intersections for a valid configuration -/
def min_intersections (config : CircleConfiguration) : ℕ :=
  (config.num_circles * config.intersections_per_circle 0) / 2

/-- Predicate for a valid circle configuration -/
def valid_configuration (config : CircleConfiguration) : Prop :=
  config.num_circles = 2008 ∧
  (∀ i, config.intersections_per_circle i ≥ 3) ∧
  config.num_intersections ≥ min_intersections config

theorem min_intersection_points (config : CircleConfiguration) 
  (h : valid_configuration config) : 
  config.num_intersections ≥ 3012 :=
sorry

end min_intersection_points_l105_10556


namespace length_of_AE_l105_10552

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AF : ℝ)
  (CE : ℝ)
  (ED : ℝ)
  (area : ℝ)

/-- Theorem stating the length of AE in the given quadrilateral -/
theorem length_of_AE (q : Quadrilateral) 
  (h1 : q.AF = 30)
  (h2 : q.CE = 40)
  (h3 : q.ED = 50)
  (h4 : q.area = 7200) : 
  ∃ AE : ℝ, AE = 322.5 := by
  sorry

end length_of_AE_l105_10552


namespace base4_division_l105_10542

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : Nat) : Base4 := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : Base4) : Base4 := sorry

/-- The theorem to be proved --/
theorem base4_division :
  divBase4 (toBase4 2023) (toBase4 13) = toBase4 155 := by sorry

end base4_division_l105_10542


namespace arithmetic_mean_reciprocals_first_four_primes_l105_10533

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l105_10533


namespace value_added_to_fraction_l105_10543

theorem value_added_to_fraction : ∀ (N V : ℝ),
  N = 8 →
  0.75 * N + V = 8 →
  V = 2 := by
sorry

end value_added_to_fraction_l105_10543


namespace six_people_arrangement_l105_10548

/-- The number of ways to arrange n people in a row -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row, where two specific people must be adjacent in a fixed order -/
def arrangements_with_fixed_pair (n : ℕ) : ℕ := (n - 1).factorial

theorem six_people_arrangement : arrangements_with_fixed_pair 6 = 120 := by sorry

end six_people_arrangement_l105_10548


namespace simplify_fraction_product_l105_10531

theorem simplify_fraction_product : 
  (3 * 5 : ℚ) / (9 * 11) * (7 * 9 * 11) / (3 * 5 * 7) = 1 := by sorry

end simplify_fraction_product_l105_10531


namespace range_of_a_l105_10581

-- Define the conditions
def p (x : ℝ) : Prop := 1 / (x - 3) ≥ 1
def q (x a : ℝ) : Prop := |x - a| < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∃ a_lower a_upper : ℝ, a_lower = 3 ∧ a_upper = 4 ∧
  ∀ a : ℝ, sufficient_not_necessary a ↔ a_lower < a ∧ a ≤ a_upper :=
sorry

end range_of_a_l105_10581


namespace vector_computation_l105_10546

theorem vector_computation :
  (4 : ℝ) • ![(-3 : ℝ), 5] - (3 : ℝ) • ![(-2 : ℝ), 6] = ![-6, 2] := by
  sorry

end vector_computation_l105_10546


namespace nine_b_equals_eighteen_l105_10500

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end nine_b_equals_eighteen_l105_10500


namespace problem_statement_l105_10597

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 1/a + 9/b = 1) 
  (h_ineq : ∀ x : ℝ, a + b ≥ -x^2 + 4*x + 18 - m) : 
  m ≥ 6 := by
  sorry

end problem_statement_l105_10597


namespace grass_field_width_l105_10593

theorem grass_field_width 
  (length : ℝ) 
  (path_width : ℝ) 
  (path_area : ℝ) 
  (h1 : length = 75) 
  (h2 : path_width = 2.8) 
  (h3 : path_area = 1518.72) : 
  ∃ width : ℝ, 
    (length + 2 * path_width) * (width + 2 * path_width) - length * width = path_area ∧ 
    width = 190.6 := by
  sorry

end grass_field_width_l105_10593


namespace f_decreasing_on_interval_l105_10566

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 11

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end f_decreasing_on_interval_l105_10566


namespace max_area_OAP_l105_10589

noncomputable section

/-- The maximum area of triangle OAP given the conditions --/
theorem max_area_OAP (a m : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : -a < m) (h4 : m ≤ (a^2 + 1)/2)
  (h5 : ∃! P : ℝ × ℝ, P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m)) :
  ∃ (S : ℝ), S = (1/54)*Real.sqrt 6 ∧ 
  (∀ A P : ℝ × ℝ, A.2 = 0 ∧ A.1^2 + a^2*A.2^2 = a^2 ∧ A.1 < 0 ∧
   P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m) →
   (1/2) * abs (A.1 * P.2) ≤ S) := by
  sorry

end

end max_area_OAP_l105_10589


namespace k_range_l105_10570

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval (5, 20)
def interval : Set ℝ := Set.Ioo 5 20

-- Define the property of having no maximum or minimum in the interval
def no_extremum (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, g y > g x ∧ ∃ z ∈ S, g z < g x

-- State the theorem
theorem k_range (k : ℝ) :
  no_extremum (f k) interval → k ∈ Set.Iic 40 ∪ Set.Ici 160 := by sorry

end k_range_l105_10570


namespace compound_propositions_truth_l105_10567

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x < y → x^2 < y^2

-- Theorem statement
theorem compound_propositions_truth :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end compound_propositions_truth_l105_10567


namespace probability_all_selected_l105_10530

theorem probability_all_selected (p_ram p_ravi p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
  sorry

end probability_all_selected_l105_10530


namespace same_color_pairs_l105_10528

def white_socks : ℕ := 5
def brown_socks : ℕ := 6
def blue_socks : ℕ := 3
def red_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem same_color_pairs : 
  (Nat.choose white_socks 2) + (Nat.choose brown_socks 2) + 
  (Nat.choose blue_socks 2) + (Nat.choose red_socks 2) = 29 := by
sorry

end same_color_pairs_l105_10528


namespace p_distance_is_300_l105_10514

/-- A race between two runners p and q -/
structure Race where
  /-- The speed of runner q in meters per second -/
  q_speed : ℝ
  /-- The length of the race course in meters -/
  race_length : ℝ

/-- The result of the race -/
def race_result (r : Race) : ℝ := 
  let p_speed := 1.2 * r.q_speed
  let p_distance := r.race_length + 50
  p_distance

/-- Theorem: Under the given conditions, p runs 300 meters -/
theorem p_distance_is_300 (r : Race) : 
  r.race_length > 0 ∧ 
  r.q_speed > 0 ∧ 
  r.race_length / r.q_speed = (r.race_length + 50) / (1.2 * r.q_speed) → 
  race_result r = 300 := by
  sorry

end p_distance_is_300_l105_10514


namespace beef_price_calculation_l105_10561

/-- The price per pound of beef, given the conditions of John's food order --/
def beef_price_per_pound : ℝ := 8

theorem beef_price_calculation (beef_amount : ℝ) (chicken_price : ℝ) (total_cost : ℝ) :
  beef_amount = 1000 →
  chicken_price = 3 →
  total_cost = 14000 →
  beef_price_per_pound * beef_amount + chicken_price * (2 * beef_amount) = total_cost := by
  sorry

#check beef_price_calculation

end beef_price_calculation_l105_10561


namespace sheila_attend_probability_l105_10555

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_attend_if_rain : ℝ := 0.3
def prob_attend_if_sunny : ℝ := 0.9
def prob_remember : ℝ := 0.9

-- Define the theorem
theorem sheila_attend_probability :
  prob_rain * prob_attend_if_rain * prob_remember +
  prob_sunny * prob_attend_if_sunny * prob_remember = 0.54 := by
  sorry

end sheila_attend_probability_l105_10555


namespace triangle_area_l105_10518

/-- Proves that a triangle with the given conditions has an original area of 4 square cm -/
theorem triangle_area (base : ℝ) (h : ℝ → ℝ) :
  h 0 = 2 →
  h 1 = h 0 + 6 →
  (1 / 2 : ℝ) * base * h 1 - (1 / 2 : ℝ) * base * h 0 = 12 →
  (1 / 2 : ℝ) * base * h 0 = 4 := by
  sorry


end triangle_area_l105_10518


namespace equilibrium_constant_is_20_l105_10526

/-- The equilibrium constant for the reaction NH₄I(s) ⇌ NH₃(g) + HI(g) -/
def equilibrium_constant (h2_conc : ℝ) (hi_conc : ℝ) : ℝ :=
  let hi_from_nh4i := hi_conc + 2 * h2_conc
  hi_from_nh4i * hi_conc

/-- Theorem stating that the equilibrium constant is 20 (mol/L)² under given conditions -/
theorem equilibrium_constant_is_20 (h2_conc : ℝ) (hi_conc : ℝ)
  (h2_eq : h2_conc = 0.5)
  (hi_eq : hi_conc = 4) :
  equilibrium_constant h2_conc hi_conc = 20 := by
  sorry

end equilibrium_constant_is_20_l105_10526


namespace perpendicular_unit_vectors_l105_10594

def a : ℝ × ℝ := (2, -2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let v₂ : ℝ × ℝ := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end perpendicular_unit_vectors_l105_10594
