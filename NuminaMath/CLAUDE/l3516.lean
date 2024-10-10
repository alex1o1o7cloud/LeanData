import Mathlib

namespace min_coefficient_value_l3516_351603

theorem min_coefficient_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box*x + 30) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∀ Box' : ℤ, (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box'*x + 30) → Box' ≥ Box) →
  Box = 61 := by
sorry

end min_coefficient_value_l3516_351603


namespace parabola_directrix_l3516_351629

/-- Given a parabola with equation y = 8x^2, its directrix has equation y = -1/32 -/
theorem parabola_directrix (x y : ℝ) :
  y = 8 * x^2 →
  ∃ (p : ℝ), p > 0 ∧ x^2 = 4 * p * y ∧ -p = -(1/32) :=
by sorry

end parabola_directrix_l3516_351629


namespace quadratic_solution_l3516_351615

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : (2 * c)^2 + c * (2 * c) + d = 0)
  (h2 : (-3 * d)^2 + c * (-3 * d) + d = 0) :
  c = -1/6 ∧ d = -1/6 := by
sorry

end quadratic_solution_l3516_351615


namespace promotion_savings_difference_l3516_351665

/-- Represents a promotion for sweater purchases -/
structure Promotion where
  first_sweater_price : ℝ
  second_sweater_discount : ℝ

/-- Calculates the total cost of two sweaters under a given promotion -/
def total_cost (p : Promotion) (original_price : ℝ) : ℝ :=
  p.first_sweater_price + (original_price - p.second_sweater_discount)

theorem promotion_savings_difference :
  let original_price : ℝ := 50
  let promotion_x : Promotion := { first_sweater_price := original_price, second_sweater_discount := 0.4 * original_price }
  let promotion_y : Promotion := { first_sweater_price := original_price, second_sweater_discount := 15 }
  total_cost promotion_y original_price - total_cost promotion_x original_price = 5 := by
  sorry

end promotion_savings_difference_l3516_351665


namespace election_vote_difference_l3516_351643

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6450 →
  candidate_percentage = 31 / 100 →
  ⌊(1 - candidate_percentage) * total_votes⌋ - ⌊candidate_percentage * total_votes⌋ = 2451 :=
by sorry

end election_vote_difference_l3516_351643


namespace complex_number_location_l3516_351628

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (3 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_location_l3516_351628


namespace permutation_count_l3516_351613

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isValidPermutation (π : Fin 10 → Fin 10) : Prop :=
  Function.Bijective π ∧
  ∀ m n : Fin 10, isPrime ((m : ℕ) + (n : ℕ)) → isPrime ((π m : ℕ) + (π n : ℕ))

theorem permutation_count :
  (∃! (count : ℕ), ∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = count ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) ∧
  (∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = 4 ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) :=
by sorry

end permutation_count_l3516_351613


namespace cos_pi_sixth_minus_alpha_l3516_351657

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/6)) 
  (h2 : Real.sin (α + π/3) = 12/13) : Real.cos (π/6 - α) = 12/13 := by
  sorry

end cos_pi_sixth_minus_alpha_l3516_351657


namespace units_digit_of_G_1009_l3516_351670

-- Define G_n
def G (n : ℕ) : ℕ := 3^(2^n) + 1

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1009 : unitsDigit (G 1009) = 4 := by
  sorry

end units_digit_of_G_1009_l3516_351670


namespace sqrt_sum_squares_l3516_351608

theorem sqrt_sum_squares : Real.sqrt (2^4 + 2^4 + 4^2) = 4 * Real.sqrt 3 := by
  sorry

end sqrt_sum_squares_l3516_351608


namespace selling_price_ratio_l3516_351692

/-- Given an item with cost price c, prove that the ratio of selling prices
    y (at 20% profit) to x (at 10% loss) is 4/3 -/
theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
  sorry

end selling_price_ratio_l3516_351692


namespace expand_product_l3516_351658

theorem expand_product (x : ℝ) : 3 * (x + 4) * (x + 5) = 3 * x^2 + 27 * x + 60 := by
  sorry

end expand_product_l3516_351658


namespace subtraction_of_fractions_l3516_351637

theorem subtraction_of_fractions : 1 / 210 - 17 / 35 = -101 / 210 := by sorry

end subtraction_of_fractions_l3516_351637


namespace function_range_in_unit_interval_l3516_351669

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) :
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
sorry

end function_range_in_unit_interval_l3516_351669


namespace diagonal_segments_100x101_l3516_351689

/-- The number of segments in the diagonal of a rectangle divided by grid lines -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - 1

/-- The width of the rectangle -/
def rectangle_width : ℕ := 100

/-- The height of the rectangle -/
def rectangle_height : ℕ := 101

theorem diagonal_segments_100x101 :
  diagonal_segments rectangle_width rectangle_height = 200 := by
  sorry

#eval diagonal_segments rectangle_width rectangle_height

end diagonal_segments_100x101_l3516_351689


namespace soda_price_ratio_l3516_351667

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_y_volume := v
  let brand_y_price := p
  let brand_z_volume := 1.3 * v
  let brand_z_price := 0.85 * p
  (brand_z_price / brand_z_volume) / (brand_y_price / brand_y_volume) = 17 / 26 := by
sorry

end soda_price_ratio_l3516_351667


namespace cone_volume_from_circle_sector_l3516_351609

/-- The volume of a right circular cone formed by rolling up a five-sixth sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let base_radius : ℝ := sector_fraction * r
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1/3) * Real.pi * base_radius^2 * height
  volume = (25/3) * Real.pi * Real.sqrt 11 := by
  sorry

end cone_volume_from_circle_sector_l3516_351609


namespace prob_red_blue_black_l3516_351695

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | White
  | Black
  | Yellow

/-- Represents a bag of marbles -/
structure MarbleBag where
  total : ℕ
  colors : List MarbleColor
  probs : MarbleColor → ℚ

/-- The probability of drawing a marble of a specific color or set of colors -/
def prob (bag : MarbleBag) (colors : List MarbleColor) : ℚ :=
  colors.map bag.probs |>.sum

/-- Theorem stating the probability of drawing a red, blue, or black marble -/
theorem prob_red_blue_black (bag : MarbleBag) :
  bag.total = 120 ∧
  bag.colors = [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
                MarbleColor.White, MarbleColor.Black, MarbleColor.Yellow] ∧
  bag.probs MarbleColor.White = 1/5 ∧
  bag.probs MarbleColor.Green = 3/10 ∧
  bag.probs MarbleColor.Yellow = 1/6 →
  prob bag [MarbleColor.Red, MarbleColor.Blue, MarbleColor.Black] = 1/3 := by
  sorry

end prob_red_blue_black_l3516_351695


namespace min_value_quadratic_roots_l3516_351683

theorem min_value_quadratic_roots (k : ℝ) (α β : ℝ) : 
  (α ^ 2 - 2 * k * α + k + 20 = 0) →
  (β ^ 2 - 2 * k * β + k + 20 = 0) →
  (k ≤ -4 ∨ k ≥ 5) →
  (∀ k', k' ≤ -4 ∨ k' ≥ 5 → (α + 1) ^ 2 + (β + 1) ^ 2 ≥ 18) ∧
  ((α + 1) ^ 2 + (β + 1) ^ 2 = 18 ↔ k = -4) :=
by sorry

end min_value_quadratic_roots_l3516_351683


namespace power_division_sum_difference_equals_sixteen_l3516_351606

theorem power_division_sum_difference_equals_sixteen :
  (5 ^ 6 / 5 ^ 4) + 3 ^ 3 - 6 ^ 2 = 16 := by
  sorry

end power_division_sum_difference_equals_sixteen_l3516_351606


namespace river_round_trip_time_l3516_351691

/-- Calculates the total time for a round trip on a river -/
theorem river_round_trip_time 
  (river_current : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_current = 8) 
  (h2 : boat_speed = 20) 
  (h3 : distance = 84) : 
  (distance / (boat_speed - river_current)) + (distance / (boat_speed + river_current)) = 10 := by
  sorry

#check river_round_trip_time

end river_round_trip_time_l3516_351691


namespace problem_1_l3516_351672

theorem problem_1 : (-2.4) + (-3.7) + (-4.6) + 5.7 = -5 := by
  sorry

#eval (-2.4) + (-3.7) + (-4.6) + 5.7

end problem_1_l3516_351672


namespace triangle_side_ratio_max_l3516_351694

theorem triangle_side_ratio_max (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (1/2) * a * b * Real.sin C = c^2 / 4 →
  (∃ (x : ℝ), a / b + b / a ≤ x) ∧ 
  (a / b + b / a ≤ 2 * Real.sqrt 2) :=
sorry

end triangle_side_ratio_max_l3516_351694


namespace calculate_savings_l3516_351655

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
  (h1 : income_ratio = 7)
  (h2 : expenditure_ratio = 6)
  (h3 : income = 14000) :
  income - (expenditure_ratio * income / income_ratio) = 2000 := by
  sorry

#check calculate_savings

end calculate_savings_l3516_351655


namespace jose_play_time_l3516_351675

/-- Calculates the total hours played given the time spent on football and basketball in minutes -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Theorem stating that playing football for 30 minutes and basketball for 60 minutes results in 1.5 hours of total play time -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end jose_play_time_l3516_351675


namespace rem_evaluation_l3516_351687

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_evaluation :
  rem (7/12 : ℚ) (-3/4 : ℚ) = -1/6 := by
  sorry

end rem_evaluation_l3516_351687


namespace derivative_at_one_l3516_351602

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_at_one :
  deriv f 1 = 3 := by sorry

end derivative_at_one_l3516_351602


namespace f_bounds_and_solution_set_l3516_351659

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds_and_solution_set :
  (∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3) ∧
  {x : ℝ | f x ≥ x^2 - 8*x + 14} = {x : ℝ | 3 ≤ x ∧ x ≤ 4 + Real.sqrt 5} :=
by sorry

end f_bounds_and_solution_set_l3516_351659


namespace three_positions_from_six_people_l3516_351674

/-- The number of ways to choose three distinct positions from a group of people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of people in the group -/
def group_size : ℕ := 6

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary 
    from a group of 6 people, where all positions must be filled by different individuals, 
    is equal to 120. -/
theorem three_positions_from_six_people : 
  choose_three_positions group_size = 120 := by sorry

end three_positions_from_six_people_l3516_351674


namespace sports_club_overlap_l3516_351653

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 3) :
  badminton + tennis - total + neither = 9 := by
  sorry

end sports_club_overlap_l3516_351653


namespace polynomial_coefficients_l3516_351676

variables (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

theorem polynomial_coefficients :
  (x + 2) * (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  a₂ = 8 ∧ a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
  sorry

end polynomial_coefficients_l3516_351676


namespace triangle_area_special_case_l3516_351622

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that under the given conditions, the area of the triangle is √15.
-/
theorem triangle_area_special_case (A B C : ℝ) (a b c : ℝ) : 
  a = 2 →
  2 * Real.sin A = Real.sin C →
  π / 2 < B → B < π →
  Real.cos (2 * C) = -1/4 →
  (1/2) * a * c * Real.sin B = Real.sqrt 15 :=
sorry

end triangle_area_special_case_l3516_351622


namespace olympiad_problem_selection_l3516_351634

theorem olympiad_problem_selection (total_initial : ℕ) (final_count : ℕ) :
  total_initial = 27 →
  final_count = 10 →
  ∃ (alina_problems masha_problems : ℕ),
    alina_problems + masha_problems = total_initial ∧
    alina_problems / 2 + 2 * masha_problems / 3 = total_initial - final_count ∧
    masha_problems - alina_problems = 15 :=
by sorry

end olympiad_problem_selection_l3516_351634


namespace integer_root_quadratic_l3516_351621

theorem integer_root_quadratic (m n : ℕ+) : 
  (∃ x : ℕ+, x^2 - (m.val * n.val) * x + (m.val + n.val) = 0) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1)) :=
by sorry

end integer_root_quadratic_l3516_351621


namespace birds_on_fence_l3516_351660

theorem birds_on_fence (initial_birds joining_birds : ℕ) :
  initial_birds + joining_birds = initial_birds + joining_birds :=
by sorry

end birds_on_fence_l3516_351660


namespace square_odd_digits_iff_one_or_three_l3516_351607

/-- A function that checks if a natural number consists of only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

/-- Theorem stating that n^2 has only odd digits if and only if n is 1 or 3 -/
theorem square_odd_digits_iff_one_or_three (n : ℕ) :
  n > 0 → (hasOnlyOddDigits (n^2) ↔ n = 1 ∨ n = 3) :=
by sorry

end square_odd_digits_iff_one_or_three_l3516_351607


namespace imaginary_part_of_z_l3516_351640

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (1 - z) = -1) :
  Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l3516_351640


namespace article_largeFont_wordsPerPage_l3516_351616

/-- Calculates the number of words per page in the large font given the article constraints. -/
def largeFont_wordsPerPage (totalWords smallFont_wordsPerPage totalPages largeFont_pages : ℕ) : ℕ :=
  let smallFont_pages := totalPages - largeFont_pages
  let smallFont_words := smallFont_pages * smallFont_wordsPerPage
  let largeFont_words := totalWords - smallFont_words
  largeFont_words / largeFont_pages

/-- Proves that the number of words per page in the large font is 1800 given the article constraints. -/
theorem article_largeFont_wordsPerPage :
  largeFont_wordsPerPage 48000 2400 21 4 = 1800 := by
  sorry

end article_largeFont_wordsPerPage_l3516_351616


namespace christine_savings_l3516_351684

/-- Calculates the amount saved given a commission rate, total sales, and personal needs allocation. -/
def amount_saved (commission_rate : ℝ) (total_sales : ℝ) (personal_needs_rate : ℝ) : ℝ :=
  let commission_earned := commission_rate * total_sales
  let savings_rate := 1 - personal_needs_rate
  savings_rate * commission_earned

/-- Proves that given a 12% commission rate on $24000 worth of sales, 
    and allocating 60% of earnings to personal needs, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved 0.12 24000 0.60 = 1152 := by
sorry

end christine_savings_l3516_351684


namespace target_circle_properties_l3516_351642

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- The given circle equation -/
def given_circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- The equation of the circle we need to prove -/
def target_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

/-- Theorem stating that the target circle passes through the intersection points
    of the line and the given circle, and also through the origin -/
theorem target_circle_properties :
  (∀ x y : ℝ, line_eq x y ∧ given_circle_eq x y → target_circle_eq x y) ∧
  target_circle_eq 0 0 := by
  sorry

end target_circle_properties_l3516_351642


namespace johns_candy_store_spending_l3516_351647

theorem johns_candy_store_spending (allowance : ℚ) :
  allowance = 4.8 →
  let arcade_spending := (3 / 5) * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 1.28 := by
sorry

end johns_candy_store_spending_l3516_351647


namespace sandra_share_l3516_351614

/-- Represents the amount of money each person receives -/
structure Share :=
  (amount : ℕ)

/-- Represents the ratio of money distribution -/
structure Ratio :=
  (sandra : ℕ)
  (amy : ℕ)
  (ruth : ℕ)

/-- Calculates the share based on the ratio and a known share -/
def calculateShare (ratio : Ratio) (knownShare : Share) (partInRatio : ℕ) : Share :=
  ⟨knownShare.amount * (ratio.sandra / partInRatio)⟩

theorem sandra_share (ratio : Ratio) (amyShare : Share) :
  ratio.sandra = 2 ∧ ratio.amy = 1 ∧ amyShare.amount = 50 →
  (calculateShare ratio amyShare ratio.amy).amount = 100 := by
  sorry

#check sandra_share

end sandra_share_l3516_351614


namespace symmetrical_letters_count_l3516_351627

-- Define a function to check if a character is symmetrical
def is_symmetrical (c : Char) : Bool :=
  c = 'A' ∨ c = 'H' ∨ c = 'I' ∨ c = 'M' ∨ c = 'O' ∨ c = 'T' ∨ c = 'U' ∨ c = 'V' ∨ c = 'W' ∨ c = 'X' ∨ c = 'Y'

-- Define the sign text
def sign_text : String := "PUNK CD FOR SALE"

-- Theorem statement
theorem symmetrical_letters_count :
  (sign_text.toList.filter is_symmetrical).length = 3 :=
sorry

end symmetrical_letters_count_l3516_351627


namespace problem_statement_l3516_351693

theorem problem_statement (x : ℝ) :
  (Real.sqrt x - 5) / 7 = 7 →
  ((x - 14)^2) / 10 = 842240.4 := by
sorry

end problem_statement_l3516_351693


namespace point_on_terminal_side_l3516_351656

theorem point_on_terminal_side (y : ℝ) (β : ℝ) : 
  (- Real.sqrt 3 : ℝ) ^ 2 + y ^ 2 > 0 →  -- Point P is not at the origin
  Real.sin β = Real.sqrt 13 / 13 →      -- Given condition for sin β
  y > 0 →                               -- y is positive (terminal side in first quadrant)
  y = 1 / 2 := by
  sorry

end point_on_terminal_side_l3516_351656


namespace bridge_length_l3516_351639

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ bridge_length : ℝ,
    bridge_length = 215 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end bridge_length_l3516_351639


namespace arithmetic_geometric_comparison_l3516_351662

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq2 : a 2 = b 2)
  (h_eq10 : a 10 = b 10) :
  a 6 > b 6 :=
sorry

end arithmetic_geometric_comparison_l3516_351662


namespace planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l3516_351619

-- Define a type for planes
variable {P : Type*}

-- Define a relation for parallelism between planes
variable (parallel : P → P → Prop)

-- Define a relation for perpendicularity between a plane and a line
variable (perpendicular : P → P → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel to each other
theorem planes_parallel_to_same_plane_are_parallel 
  (p1 p2 p3 : P) 
  (h1 : parallel p1 p3) 
  (h2 : parallel p2 p3) : 
  parallel p1 p2 := by sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel to each other
theorem planes_perpendicular_to_same_line_are_parallel 
  (p1 p2 l : P) 
  (h1 : perpendicular p1 l) 
  (h2 : perpendicular p2 l) : 
  parallel p1 p2 := by sorry

end planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l3516_351619


namespace merchant_problem_l3516_351644

theorem merchant_problem (n : ℕ) : 
  (100 * n^2 : ℕ) / 100 * (2 * n) = 2662 → n = 11 := by
  sorry

end merchant_problem_l3516_351644


namespace extraordinary_stack_size_l3516_351605

/-- An extraordinary stack of cards -/
structure ExtraordinaryStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a_size : ℕ := n)
  (pile_b_size : ℕ := n)
  (card_57_from_a_position : ℕ := 57)
  (card_200_from_b_position : ℕ := 200)

/-- The number of cards in an extraordinary stack is 198 -/
theorem extraordinary_stack_size :
  ∀ (stack : ExtraordinaryStack),
    stack.card_57_from_a_position % 2 = 1 →
    stack.card_200_from_b_position % 2 = 0 →
    stack.card_57_from_a_position ≤ stack.total_cards →
    stack.card_200_from_b_position ≤ stack.total_cards →
    stack.total_cards = 198 := by
  sorry

end extraordinary_stack_size_l3516_351605


namespace expression_evaluation_l3516_351636

theorem expression_evaluation (a b : ℝ) (ha : a = 6) (hb : b = 2) :
  3 / (a + b) + a^2 = 291 / 8 := by sorry

end expression_evaluation_l3516_351636


namespace pat_to_mark_ratio_l3516_351666

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Conditions of the problem --/
def project_conditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 180 ∧
  h.pat = 2 * h.kate ∧
  h.mark = h.kate + 100

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (h : ProjectHours) :
  project_conditions h → h.pat * 3 = h.mark * 1 := by
  sorry

#check pat_to_mark_ratio

end pat_to_mark_ratio_l3516_351666


namespace part_time_employees_l3516_351673

/-- Represents the number of employees in a corporation -/
structure Corporation where
  total : ℕ
  fullTime : ℕ
  partTime : ℕ

/-- The total number of employees is the sum of full-time and part-time employees -/
axiom total_eq_sum (c : Corporation) : c.total = c.fullTime + c.partTime

/-- Theorem: Given a corporation with 65,134 total employees and 63,093 full-time employees,
    the number of part-time employees is 2,041 -/
theorem part_time_employees (c : Corporation) 
    (h1 : c.total = 65134) 
    (h2 : c.fullTime = 63093) : 
    c.partTime = 2041 := by
  sorry


end part_time_employees_l3516_351673


namespace original_bill_calculation_l3516_351678

theorem original_bill_calculation (num_friends : ℕ) (discount_rate : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  discount_rate = 6 / 100 →
  individual_payment = 188 / 10 →
  ∃ (original_bill : ℚ), 
    (1 - discount_rate) * original_bill = num_friends * individual_payment ∧
    original_bill = 100 := by
  sorry

end original_bill_calculation_l3516_351678


namespace part_one_part_two_part_three_l3516_351661

-- Define the function f and its properties
def f (x : ℝ) : ℝ := sorry

-- Assume |f'(x)| < 1 for all x in the domain of f
axiom f_deriv_bound (x : ℝ) : |deriv f x| < 1

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x ∈ Set.Icc 1 2, f x = a * x + Real.log x) :
  a ∈ Set.Ioo (-3/2) 0 := sorry

-- Part 2
theorem part_two : ∃! x, f x = x := sorry

-- Part 3
def is_periodic (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem part_three (h : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| < 1 := sorry

end part_one_part_two_part_three_l3516_351661


namespace smallest_norm_given_condition_l3516_351626

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that the norm of v + (4, 2) is 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
  (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
  ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end smallest_norm_given_condition_l3516_351626


namespace segment_length_l3516_351631

/-- The length of a segment with endpoints (1,1) and (8,17) is √305 -/
theorem segment_length : Real.sqrt ((8 - 1)^2 + (17 - 1)^2) = Real.sqrt 305 := by
  sorry

end segment_length_l3516_351631


namespace complex_power_sum_l3516_351646

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 3 * i^23 + 2 * i^47 = -5 * i := by
  sorry

end complex_power_sum_l3516_351646


namespace triangle_formation_l3516_351652

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  triangle_inequality 2 3 4 :=
sorry

end triangle_formation_l3516_351652


namespace triangle_property_l3516_351625

/-- Given a triangle ABC with angles A, B, C satisfying the given condition,
    prove that A = π/3 and the maximum area is 3√3/4 when the circumradius is 1 -/
theorem triangle_property (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.sin A - Real.sin B + Real.sin C) / Real.sin C = 
        Real.sin B / (Real.sin A + Real.sin B - Real.sin C)) :
  A = π/3 ∧ 
  (∀ S : Real, S ≤ 3 * Real.sqrt 3 / 4 ∧ 
    ∃ a b c : Real, 0 < a ∧ 0 < b ∧ 0 < c ∧
      a^2 + b^2 + c^2 = 2 * (a*b + b*c + c*a) ∧
      S = (Real.sin A * b * c) / 2) := by
  sorry

end triangle_property_l3516_351625


namespace non_monotonic_interval_implies_k_range_l3516_351611

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), a < x ∧ x < y ∧ y < z ∧ z < b ∧
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → 1 ≤ k ∧ k < 3/2 := by sorry

end non_monotonic_interval_implies_k_range_l3516_351611


namespace ludwig_earnings_l3516_351641

/-- Calculates the weekly earnings of a worker with given work schedule and daily salary. -/
def weeklyEarnings (totalDays : ℕ) (halfDays : ℕ) (dailySalary : ℚ) : ℚ :=
  let fullDays := totalDays - halfDays
  fullDays * dailySalary + halfDays * (dailySalary / 2)

/-- Theorem stating that under the given conditions, the weekly earnings are $55. -/
theorem ludwig_earnings :
  weeklyEarnings 7 3 10 = 55 := by
  sorry

end ludwig_earnings_l3516_351641


namespace midpoint_coordinate_product_l3516_351668

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (3, -4) and (5, -8) is -24. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := -8
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = -24 := by
  sorry

end midpoint_coordinate_product_l3516_351668


namespace man_rowing_speed_l3516_351690

/-- The speed of a man rowing a boat against the stream, given his speed with the stream and his rate in still water. -/
def speed_against_stream (speed_with_stream : ℝ) (rate_still_water : ℝ) : ℝ :=
  abs (2 * rate_still_water - speed_with_stream)

/-- Theorem: Given a man's speed with the stream of 22 km/h and his rate in still water of 6 km/h, his speed against the stream is 10 km/h. -/
theorem man_rowing_speed 
  (h1 : speed_with_stream = 22)
  (h2 : rate_still_water = 6) :
  speed_against_stream speed_with_stream rate_still_water = 10 := by
  sorry

#eval speed_against_stream 22 6

end man_rowing_speed_l3516_351690


namespace unique_abcd_l3516_351677

def is_valid_abcd (abcd : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    abcd = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 < d ∧ d < 10 ∧
    abcd / d = d * 100 + b * 10 + a

theorem unique_abcd :
  ∃! abcd : ℕ, is_valid_abcd abcd ∧ abcd = 1964 :=
sorry

end unique_abcd_l3516_351677


namespace repeating_decimal_fraction_sum_l3516_351624

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end repeating_decimal_fraction_sum_l3516_351624


namespace playground_area_not_covered_l3516_351650

theorem playground_area_not_covered (playground_side : ℝ) (building_length building_width : ℝ) : 
  playground_side = 12 →
  building_length = 8 →
  building_width = 5 →
  playground_side * playground_side - building_length * building_width = 104 := by
sorry

end playground_area_not_covered_l3516_351650


namespace expression_evaluation_l3516_351654

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end expression_evaluation_l3516_351654


namespace torn_sheets_count_l3516_351612

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the last page in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The number of the first torn out page -/
  first : ℕ
  /-- The number of the last torn out page -/
  last : ℕ

/-- Check if a number consists of the same digits as another number in a different order -/
def sameDigitsDifferentOrder (a b : ℕ) : Prop :=
  sorry

/-- Calculate the number of sheets torn out given the first and last torn page numbers -/
def sheetsTornOut (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem to be proved -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 ∧
  sameDigitsDifferentOrder torn.first torn.last ∧
  Even torn.last ∧
  torn.last > torn.first →
  sheetsTornOut torn = 167 := by
  sorry

end torn_sheets_count_l3516_351612


namespace parabola_c_value_l3516_351617

/-- A parabola is defined by the equation x = ay² + by + c, where a, b, and c are constants -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℚ) : ℚ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-5) = 1 → p.x_coord (-1) = 4 → p.c = 145/12 := by
  sorry

end parabola_c_value_l3516_351617


namespace pencil_cost_l3516_351635

-- Define the cost of a pen and a pencil in cents
variable (p q : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 3 * p + 4 * q = 287
def condition2 : Prop := 5 * p + 2 * q = 236

-- Theorem to prove
theorem pencil_cost (h1 : condition1 p q) (h2 : condition2 p q) : q = 52 := by
  sorry

end pencil_cost_l3516_351635


namespace class_composition_l3516_351681

/-- Represents a pair of numbers written by a child -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid given the actual number of boys and girls -/
def is_valid_response (r : Response) (actual_boys : ℕ) (actual_girls : ℕ) : Prop :=
  (r.boys = actual_boys - 1 ∧ (r.girls = actual_girls - 1 + 4 ∨ r.girls = actual_girls - 1 - 4)) ∨
  (r.girls = actual_girls - 1 ∧ (r.boys = actual_boys - 1 + 4 ∨ r.boys = actual_boys - 1 - 4))

/-- The theorem to be proved -/
theorem class_composition :
  ∃ (boys girls : ℕ),
    boys = 14 ∧ girls = 15 ∧
    is_valid_response ⟨10, 14⟩ boys girls ∧
    is_valid_response ⟨13, 11⟩ boys girls ∧
    is_valid_response ⟨13, 19⟩ boys girls ∧
    ∀ (b g : ℕ),
      (is_valid_response ⟨10, 14⟩ b g ∧
       is_valid_response ⟨13, 11⟩ b g ∧
       is_valid_response ⟨13, 19⟩ b g) →
      b = boys ∧ g = girls :=
sorry

end class_composition_l3516_351681


namespace geometric_sequence_problem_l3516_351663

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 3 = 2 * a 1 →             -- given condition
  (a 4 + 2 * a 7) / 2 = 5 / 4 →     -- given condition
  q = 1 / 2 := by
sorry

end geometric_sequence_problem_l3516_351663


namespace tea_pot_volume_l3516_351600

/-- The amount of tea in milliliters per cup -/
def tea_per_cup : ℕ := 65

/-- The number of cups filled with tea -/
def cups_filled : ℕ := 16

/-- The total amount of tea in the pot in milliliters -/
def total_tea : ℕ := tea_per_cup * cups_filled

/-- Theorem stating that the total amount of tea in the pot is 1040 ml -/
theorem tea_pot_volume : total_tea = 1040 := by
  sorry

end tea_pot_volume_l3516_351600


namespace quadratic_inequality_l3516_351686

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 < 0 ↔ -6 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_l3516_351686


namespace units_digit_sum_l3516_351698

theorem units_digit_sum (n : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end units_digit_sum_l3516_351698


namespace no_prime_sqrt_sum_integer_l3516_351632

theorem no_prime_sqrt_sum_integer :
  ¬ ∃ (p n : ℕ), Prime p ∧ n > 0 ∧ ∃ (k : ℤ), (Int.sqrt (p + n) + Int.sqrt n : ℤ) = k :=
sorry

end no_prime_sqrt_sum_integer_l3516_351632


namespace total_junk_mail_l3516_351623

/-- Given a block with houses and junk mail distribution, calculate the total junk mail. -/
theorem total_junk_mail (num_houses : ℕ) (mail_per_house : ℕ) : num_houses = 10 → mail_per_house = 35 → num_houses * mail_per_house = 350 := by
  sorry

end total_junk_mail_l3516_351623


namespace min_value_expression_l3516_351648

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1 / (a + b)^3 ≥ 1 / (4^(1/5 : ℝ)) :=
by sorry

end min_value_expression_l3516_351648


namespace ellipse_eccentricity_l3516_351685

theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  c^2 = a^2 - b^2 →
  c^2 = m^2 + n^2 →
  c^2 = a * m →
  n^2 = (2 * m^2 + c^2) / 2 →
  c / a = 1 / 2 := by
sorry

end ellipse_eccentricity_l3516_351685


namespace max_value_of_S_l3516_351682

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    (S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end max_value_of_S_l3516_351682


namespace minimum_value_theorem_l3516_351618

theorem minimum_value_theorem (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∀ ε > 0, ∃ x : ℝ, (x^2 + 5) / Real.sqrt (x^2 + 4) < 5/2 + ε :=
by sorry

end minimum_value_theorem_l3516_351618


namespace share_of_A_l3516_351638

theorem share_of_A (total : ℝ) (a b c : ℝ) : 
  total = 116000 →
  a + b + c = total →
  a / b = 3 / 4 →
  b / c = 5 / 6 →
  a = 116000 * 15 / 59 :=
by sorry

end share_of_A_l3516_351638


namespace richs_walk_total_distance_l3516_351651

/-- Calculates the total distance of Rich's walk --/
def richs_walk (segment1 segment2 segment5 : ℝ) : ℝ :=
  let segment3 := 2 * (segment1 + segment2)
  let segment4 := 1.5 * segment3
  let sum_to_5 := segment1 + segment2 + segment3 + segment4 + segment5
  let segment6 := 3 * sum_to_5
  let sum_to_6 := sum_to_5 + segment6
  let segment7 := 0.75 * sum_to_6
  let one_way := segment1 + segment2 + segment3 + segment4 + segment5 + segment6 + segment7
  2 * one_way

theorem richs_walk_total_distance :
  richs_walk 20 200 300 = 22680 := by
  sorry

end richs_walk_total_distance_l3516_351651


namespace function_sum_equals_four_l3516_351697

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem function_sum_equals_four (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m →
  f 5 + f (-5) = 4 := by
  sorry

end function_sum_equals_four_l3516_351697


namespace church_full_capacity_l3516_351688

/-- Calculates the total number of people that can be seated in a church with three sections -/
def church_capacity (section1_rows section1_chairs_per_row section2_rows section2_chairs_per_row section3_rows section3_chairs_per_row : ℕ) : ℕ :=
  section1_rows * section1_chairs_per_row +
  section2_rows * section2_chairs_per_row +
  section3_rows * section3_chairs_per_row

/-- Theorem stating that the church capacity is 490 given the specified section configurations -/
theorem church_full_capacity :
  church_capacity 15 8 20 6 25 10 = 490 := by
  sorry

end church_full_capacity_l3516_351688


namespace complex_power_one_minus_i_six_l3516_351671

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_one_minus_i_six :
  (1 - i)^6 = 8*i := by sorry

end complex_power_one_minus_i_six_l3516_351671


namespace min_value_2a_plus_1_l3516_351679

theorem min_value_2a_plus_1 (a : ℝ) (h : 6 * a^2 + 5 * a + 4 = 3) :
  ∃ (m : ℝ), (2 * a + 1 ≥ m) ∧ (∀ x, 6 * x^2 + 5 * x + 4 = 3 → 2 * x + 1 ≥ m) ∧ m = 0 :=
sorry

end min_value_2a_plus_1_l3516_351679


namespace august_has_five_tuesdays_l3516_351645

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def count_days (m : Month) : DayOfWeek → Nat :=
  sorry

/-- Returns true if the given day occurs exactly five times in the month -/
def occurs_five_times (d : DayOfWeek) (m : Month) : Prop :=
  count_days m d = 5

/-- Theorem: If July has five Fridays, then August must have five Tuesdays -/
theorem august_has_five_tuesdays
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : occurs_five_times DayOfWeek.Friday july) :
  occurs_five_times DayOfWeek.Tuesday august :=
sorry

end august_has_five_tuesdays_l3516_351645


namespace gum_per_nickel_l3516_351680

/-- 
Given:
- initial_nickels: The number of nickels Quentavious started with
- remaining_nickels: The number of nickels Quentavious had left
- total_gum: The total number of gum pieces Quentavious received

Prove: The number of gum pieces per nickel is 2
-/
theorem gum_per_nickel 
  (initial_nickels : ℕ) 
  (remaining_nickels : ℕ) 
  (total_gum : ℕ) 
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum = 6)
  : (total_gum : ℚ) / (initial_nickels - remaining_nickels : ℚ) = 2 := by
  sorry

end gum_per_nickel_l3516_351680


namespace total_laundry_cost_l3516_351649

def laundry_cost (washer_cost : ℝ) (dryer_cost_per_10_min : ℝ) (loads : ℕ) 
  (special_soap_cost : ℝ) (num_dryers : ℕ) (dryer_time : ℕ) (membership_fee : ℝ) : ℝ :=
  let washing_cost := washer_cost * loads + special_soap_cost
  let dryer_cost := (↑num_dryers * ↑(dryer_time / 10 + 1)) * dryer_cost_per_10_min
  washing_cost + dryer_cost + membership_fee

theorem total_laundry_cost :
  laundry_cost 4 0.25 3 2.5 4 45 10 = 29.5 := by
  sorry

end total_laundry_cost_l3516_351649


namespace goods_train_speed_l3516_351664

/-- The speed of a goods train crossing a platform -/
theorem goods_train_speed (platform_length : ℝ) (crossing_time : ℝ) (train_length : ℝ)
  (h1 : platform_length = 250)
  (h2 : crossing_time = 26)
  (h3 : train_length = 270.0416) :
  ∃ (speed : ℝ), abs (speed - 20) < 0.01 ∧ 
  speed = (platform_length + train_length) / crossing_time :=
sorry

end goods_train_speed_l3516_351664


namespace prime_characterization_l3516_351601

theorem prime_characterization (p : ℕ) (h1 : p > 3) (h2 : (p^2 + 15) % 12 = 4) :
  Nat.Prime p :=
sorry

end prime_characterization_l3516_351601


namespace final_eraser_count_l3516_351630

/-- Represents the state of erasers in three drawers -/
structure EraserState where
  drawer1 : ℕ
  drawer2 : ℕ
  drawer3 : ℕ

/-- Initial state of erasers -/
def initial_state : EraserState := ⟨139, 95, 75⟩

/-- State after Monday's changes -/
def monday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 50, s.drawer2 - 50, s.drawer3⟩

/-- State after Tuesday's changes -/
def tuesday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 - 35, s.drawer2, s.drawer3 - 20⟩

/-- Final state after changes later in the week -/
def final_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 131, s.drawer2 - 30, s.drawer3⟩

/-- Total number of erasers in all drawers -/
def total_erasers (s : EraserState) : ℕ :=
  s.drawer1 + s.drawer2 + s.drawer3

/-- Theorem stating the final number of erasers -/
theorem final_eraser_count :
  total_erasers (final_state (tuesday_state (monday_state initial_state))) = 355 := by
  sorry


end final_eraser_count_l3516_351630


namespace fifty_eight_prime_sum_l3516_351610

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n + 1))).card

/-- Theorem stating that 58 can be written as the sum of two primes in exactly 3 ways -/
theorem fifty_eight_prime_sum : count_prime_pairs 58 = 3 := by
  sorry

end fifty_eight_prime_sum_l3516_351610


namespace expression_value_l3516_351699

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by sorry

end expression_value_l3516_351699


namespace keychain_arrangements_l3516_351620

/-- The number of keys on the keychain -/
def total_keys : ℕ := 7

/-- The number of distinct arrangements of keys on a keychain,
    where two specific keys must be adjacent and arrangements
    are considered identical under rotation and reflection -/
def distinct_arrangements : ℕ := 60

/-- Theorem stating that the number of distinct arrangements
    of keys on the keychain is equal to 60 -/
theorem keychain_arrangements :
  (total_keys : ℕ) = 7 →
  distinct_arrangements = 60 := by
  sorry

end keychain_arrangements_l3516_351620


namespace abs_neg_two_equals_two_l3516_351604

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := by
  sorry

end abs_neg_two_equals_two_l3516_351604


namespace absolute_value_inequality_l3516_351633

theorem absolute_value_inequality (x : ℝ) :
  |6 - x| / 4 > 1 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 10 := by
  sorry

end absolute_value_inequality_l3516_351633


namespace system_two_solutions_l3516_351696

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 169 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (s : Set (ℝ × ℝ)), s.ncard = 2 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ 
      (|x + y + 5| + |y - x + 5| = 10 ∧
       (|x| - 12)^2 + (|y| - 5)^2 = a))) ↔
  (a = 49 ∨ a = 169) :=
sorry

end system_two_solutions_l3516_351696
