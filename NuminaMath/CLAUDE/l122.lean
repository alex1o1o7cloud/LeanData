import Mathlib

namespace max_value_sum_of_roots_l122_12213

theorem max_value_sum_of_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) :
  (∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
    Real.sqrt (3 * x^2 + 1) + Real.sqrt (3 * y^2 + 1) + Real.sqrt (3 * z^2 + 1) > 
    Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1)) ∨
  (Real.sqrt (3 * a^2 + 1) + Real.sqrt (3 * b^2 + 1) + Real.sqrt (3 * c^2 + 1) = Real.sqrt 201) :=
by sorry

end max_value_sum_of_roots_l122_12213


namespace flowers_purchase_l122_12222

theorem flowers_purchase (dozen_bought : ℕ) : 
  (∀ d : ℕ, 12 * d + 2 * d = 14 * d) →
  12 * dozen_bought + 2 * dozen_bought = 42 →
  dozen_bought = 3 := by
  sorry

end flowers_purchase_l122_12222


namespace runners_meeting_time_l122_12215

def lap_time_bob : ℕ := 8
def lap_time_carol : ℕ := 9
def lap_time_ted : ℕ := 10

def meeting_time : ℕ := 360

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm lap_time_bob lap_time_carol) lap_time_ted = meeting_time :=
by sorry

end runners_meeting_time_l122_12215


namespace inverse_difference_l122_12297

theorem inverse_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y + 1) :
  1 / x - 1 / y = -1 - 1 / (x * y) := by
  sorry

end inverse_difference_l122_12297


namespace grasshopper_jump_distance_l122_12212

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  frog_jump : ℕ
  frog_grasshopper_diff : ℕ
  frog_mouse_diff : ℕ

/-- Theorem stating the grasshopper's jump distance given the conditions -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h1 : contest.frog_jump = 58)
  (h2 : contest.frog_grasshopper_diff = 39) :
  contest.frog_jump - contest.frog_grasshopper_diff = 19 := by
  sorry

#check grasshopper_jump_distance

end grasshopper_jump_distance_l122_12212


namespace picture_book_shelves_l122_12214

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
  sorry

end picture_book_shelves_l122_12214


namespace average_of_25_results_l122_12287

theorem average_of_25_results (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : (results.take 12).sum / 12 = 14)
  (h3 : (results.drop 13).sum / 12 = 17)
  (h4 : results[12] = 128) :
  results.sum / 25 = 20 := by
  sorry

end average_of_25_results_l122_12287


namespace x_squared_minus_y_squared_l122_12246

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/11) 
  (h2 : x - y = 1/55) : 
  x^2 - y^2 = 1/121 := by
sorry

end x_squared_minus_y_squared_l122_12246


namespace quadratic_roots_property_l122_12248

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3 * a - 4) * (2 * b - 2) = -4 := by
sorry

end quadratic_roots_property_l122_12248


namespace log_expression_equals_negative_one_l122_12219

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by sorry

end log_expression_equals_negative_one_l122_12219


namespace unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l122_12230

theorem unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2 :
  ∃! n : ℕ+, 24 ∣ n ∧ 8 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 8.2 :=
by sorry

end unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l122_12230


namespace biker_bob_distance_l122_12272

/-- The total distance covered by Biker Bob between town A and town B -/
def total_distance : ℝ := 155

/-- The distance of the first segment (west) -/
def distance_west : ℝ := 45

/-- The distance of the second segment (northwest) -/
def distance_northwest : ℝ := 25

/-- The distance of the third segment (south) -/
def distance_south : ℝ := 35

/-- The distance of the fourth segment (east) -/
def distance_east : ℝ := 50

/-- Theorem stating that the total distance is the sum of all segment distances -/
theorem biker_bob_distance : 
  total_distance = distance_west + distance_northwest + distance_south + distance_east := by
  sorry

#check biker_bob_distance

end biker_bob_distance_l122_12272


namespace count_numbers_satisfying_condition_l122_12254

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the property we're looking for
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧ n = 7 * sumOfDigits n

-- State the theorem
theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), S.card = 3 ∧ ∀ n, n ∈ S ↔ satisfiesCondition n :=
sorry

end count_numbers_satisfying_condition_l122_12254


namespace simple_interest_time_l122_12249

/-- Simple interest calculation -/
theorem simple_interest_time (principal rate interest : ℝ) :
  principal > 0 →
  rate > 0 →
  interest > 0 →
  (interest * 100) / (principal * rate) = 2 →
  principal = 400 →
  rate = 12.5 →
  interest = 100 →
  (interest * 100) / (principal * rate) = 2 := by
  sorry

end simple_interest_time_l122_12249


namespace chocolate_distribution_l122_12226

/-- Calculates the number of chocolate squares each student receives when:
  * Gerald brings 7 chocolate bars
  * Each bar contains 8 squares
  * For every bar Gerald brings, the teacher brings 2 more identical ones
  * There are 24 students in class
-/
theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_ratio : Nat) (num_students : Nat)
    (h1 : gerald_bars = 7)
    (h2 : squares_per_bar = 8)
    (h3 : teacher_ratio = 2)
    (h4 : num_students = 24) :
    (gerald_bars + gerald_bars * teacher_ratio) * squares_per_bar / num_students = 7 := by
  sorry


end chocolate_distribution_l122_12226


namespace cell_division_chromosome_count_l122_12263

/-- Represents the number of chromosomes in a fruit fly cell -/
def ChromosomeCount : ℕ := 8

/-- Represents the possible chromosome counts during cell division -/
def PossibleChromosomeCounts : Set ℕ := {8, 16}

/-- Represents a genotype with four alleles -/
structure Genotype :=
  (allele1 allele2 allele3 allele4 : Char)

/-- Represents a fruit fly cell -/
structure FruitFlyCell :=
  (genotype : Genotype)
  (chromosomeCount : ℕ)

/-- Axiom: Fruit flies have 2N=8 chromosomes -/
axiom fruit_fly_chromosome_count : ChromosomeCount = 8

/-- Axiom: Alleles A/a and B/b are inherited independently -/
axiom alleles_independent : True

/-- Theorem: A fruit fly cell with genotype AAaaBBbb during cell division
    contains either 8 or 16 chromosomes -/
theorem cell_division_chromosome_count
  (cell : FruitFlyCell)
  (h_genotype : cell.genotype = ⟨'A', 'A', 'B', 'B'⟩ ∨
                cell.genotype = ⟨'a', 'a', 'b', 'b'⟩) :
  cell.chromosomeCount ∈ PossibleChromosomeCounts := by
  sorry

end cell_division_chromosome_count_l122_12263


namespace max_running_speed_l122_12276

/-- The maximum speed at which a person can run to catch a train, given specific conditions -/
theorem max_running_speed (x : ℝ) (h : x > 0) : 
  let v := (30 : ℝ) / 3
  let train_speed := (30 : ℝ)
  let distance_fraction := (1 : ℝ) / 3
  (distance_fraction * x) / v = x / train_speed ∧ 
  ((1 - distance_fraction) * x) / v = (x + (distance_fraction * x)) / train_speed →
  v = 10 := by
  sorry

end max_running_speed_l122_12276


namespace farm_has_eleven_goats_l122_12291

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the conditions of the farm -/
def valid_farm (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.goats + f.cows + f.pigs = 56

/-- Theorem stating that a valid farm has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : valid_farm f) : f.goats = 11 := by
  sorry

#check farm_has_eleven_goats

end farm_has_eleven_goats_l122_12291


namespace min_value_problem_l122_12265

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 :=
sorry

end min_value_problem_l122_12265


namespace f_decreasing_implies_a_range_l122_12274

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end

end f_decreasing_implies_a_range_l122_12274


namespace youngest_sibling_age_l122_12228

/-- The age of the youngest sibling in a family of 6 siblings -/
def youngest_age : ℝ := 17.5

/-- The number of siblings in the family -/
def num_siblings : ℕ := 6

/-- The age differences between the siblings and the youngest sibling -/
def age_differences : List ℝ := [4, 5, 7, 9, 11]

/-- The average age of all siblings -/
def average_age : ℝ := 23.5

/-- Theorem stating that given the conditions, the age of the youngest sibling is 17.5 -/
theorem youngest_sibling_age :
  let ages := youngest_age :: (age_differences.map (· + youngest_age))
  (ages.sum / num_siblings) = average_age ∧
  ages.length = num_siblings :=
by sorry

end youngest_sibling_age_l122_12228


namespace prime_product_divisors_l122_12258

theorem prime_product_divisors (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → (Finset.card (Nat.divisors (p^n * q^7)) = 56) → n = 6 := by
  sorry

end prime_product_divisors_l122_12258


namespace y_derivative_l122_12239

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 - 3*x - 2*x^2) + (3 / (2 * Real.sqrt 2)) * Real.arcsin ((4*x + 3) / Real.sqrt 17)

theorem y_derivative (x : ℝ) : 
  deriv y x = -(2*x) / Real.sqrt (1 - 3*x - 2*x^2) :=
by sorry

end y_derivative_l122_12239


namespace vip_price_is_60_l122_12218

/-- Represents the ticket sales and pricing for a snooker tournament --/
structure SnookerTickets where
  totalTickets : ℕ
  totalRevenue : ℕ
  generalPrice : ℕ
  vipDifference : ℕ

/-- The specific ticket sales scenario for the tournament --/
def tournamentSales : SnookerTickets :=
  { totalTickets := 320
  , totalRevenue := 7500
  , generalPrice := 10
  , vipDifference := 148
  }

/-- Calculates the price of a VIP ticket --/
def vipPrice (s : SnookerTickets) : ℕ :=
  let generalTickets := (s.totalTickets + s.vipDifference) / 2
  let vipTickets := s.totalTickets - generalTickets
  (s.totalRevenue - s.generalPrice * generalTickets) / vipTickets

/-- Theorem stating that the VIP ticket price for the given scenario is $60 --/
theorem vip_price_is_60 : vipPrice tournamentSales = 60 := by
  sorry

end vip_price_is_60_l122_12218


namespace cucumbers_per_kind_paulines_garden_cucumbers_l122_12284

/-- Calculates the number of cucumbers of each kind in Pauline's garden. -/
theorem cucumbers_per_kind (total_spaces : ℕ) (total_tomatoes : ℕ) (total_potatoes : ℕ) 
  (cucumber_kinds : ℕ) (empty_spaces : ℕ) : ℕ :=
  by
  have filled_spaces : ℕ := total_spaces - empty_spaces
  have non_cucumber_spaces : ℕ := total_tomatoes + total_potatoes
  have cucumber_spaces : ℕ := filled_spaces - non_cucumber_spaces
  exact cucumber_spaces / cucumber_kinds

/-- Proves that Pauline has planted 4 cucumbers of each kind in her garden. -/
theorem paulines_garden_cucumbers : 
  cucumbers_per_kind 150 15 30 5 85 = 4 :=
by sorry

end cucumbers_per_kind_paulines_garden_cucumbers_l122_12284


namespace specific_polyhedron_space_diagonals_l122_12241

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 
  (2 * Q.quadrilateral_faces + 5 * Q.pentagon_faces)

/-- Theorem: A specific convex polyhedron Q has 321 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 42 ∧
    Q.triangular_faces = 26 ∧
    Q.quadrilateral_faces = 12 ∧
    Q.pentagon_faces = 4 ∧
    space_diagonals Q = 321 := by
  sorry

end specific_polyhedron_space_diagonals_l122_12241


namespace triangular_number_all_equal_digits_l122_12266

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_digits_equal (num : ℕ) (digit : ℕ) : Prop :=
  ∀ d, d ∈ num.digits 10 → d = digit

theorem triangular_number_all_equal_digits :
  {a : ℕ | a < 10 ∧ ∃ n : ℕ, n ≥ 4 ∧ all_digits_equal (triangular_number n) a} = {5, 6} := by
  sorry

end triangular_number_all_equal_digits_l122_12266


namespace jennis_age_l122_12286

theorem jennis_age (sum difference : ℕ) (h1 : sum = 70) (h2 : difference = 32) :
  ∃ (mrs_bai jenni : ℕ), mrs_bai + jenni = sum ∧ mrs_bai - jenni = difference ∧ jenni = 19 := by
  sorry

end jennis_age_l122_12286


namespace two_thousand_twelfth_digit_l122_12207

def digit_sequence (n : ℕ) : ℕ :=
  sorry

theorem two_thousand_twelfth_digit :
  digit_sequence 2012 = 0 :=
sorry

end two_thousand_twelfth_digit_l122_12207


namespace angle_relationships_l122_12227

theorem angle_relationships (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  C = B / 2 →    -- C is half of B
  A = 6 * B →    -- A is 6 times B
  (A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7) := by
  sorry

end angle_relationships_l122_12227


namespace gain_percentage_calculation_l122_12261

/-- Given a selling price and a gain, calculate the gain percentage. -/
theorem gain_percentage_calculation (selling_price gain : ℝ) 
  (h1 : selling_price = 195)
  (h2 : gain = 45) : 
  (gain / (selling_price - gain)) * 100 = 30 := by
  sorry

end gain_percentage_calculation_l122_12261


namespace claudia_weekend_earnings_l122_12282

/-- Calculates the total earnings from weekend art classes -/
def weekend_earnings (cost_per_class : ℚ) (saturday_attendees : ℕ) : ℚ :=
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  cost_per_class * total_attendees

/-- Proves that Claudia's total earnings from her weekend art classes are $300.00 -/
theorem claudia_weekend_earnings :
  weekend_earnings 10 20 = 300 := by
  sorry

end claudia_weekend_earnings_l122_12282


namespace imaginary_part_of_complex_fraction_l122_12257

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.im ((1 + i) / (1 - i)) = 1 := by sorry

end imaginary_part_of_complex_fraction_l122_12257


namespace pages_per_side_l122_12216

/-- Given the conditions of James' printing job, prove the number of pages per side. -/
theorem pages_per_side (num_books : ℕ) (pages_per_book : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_book = 600 → 
  num_sheets = 150 → 
  (num_books * pages_per_book) / (num_sheets * 2) = 4 := by
  sorry

end pages_per_side_l122_12216


namespace salary_calculation_l122_12269

theorem salary_calculation (S : ℝ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * (3 / 5) = S * (3 / 5))
  (remaining : S - (S * (1 / 5) + S * (1 / 10) + S * (3 / 5)) = 16000) :
  S = 160000 := by
  sorry

end salary_calculation_l122_12269


namespace paula_karl_age_problem_l122_12232

/-- Represents the ages and time in the problem about Paula and Karl --/
structure AgesProblem where
  paula_age : ℕ
  karl_age : ℕ
  years_until_double : ℕ

/-- The conditions of the problem are satisfied --/
def satisfies_conditions (ap : AgesProblem) : Prop :=
  (ap.paula_age - 5 = 3 * (ap.karl_age - 5)) ∧
  (ap.paula_age + ap.karl_age = 54) ∧
  (ap.paula_age + ap.years_until_double = 2 * (ap.karl_age + ap.years_until_double))

/-- The theorem stating that the solution to the problem is 6 years --/
theorem paula_karl_age_problem :
  ∃ (ap : AgesProblem), satisfies_conditions ap ∧ ap.years_until_double = 6 :=
by sorry

end paula_karl_age_problem_l122_12232


namespace melanie_catch_melanie_catch_is_ten_l122_12278

def sara_catch : ℕ := 5
def melanie_multiplier : ℕ := 2

theorem melanie_catch : ℕ := sara_catch * melanie_multiplier

theorem melanie_catch_is_ten : melanie_catch = 10 := by
  sorry

end melanie_catch_melanie_catch_is_ten_l122_12278


namespace modulus_of_5_minus_12i_l122_12233

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end modulus_of_5_minus_12i_l122_12233


namespace action_figure_cost_l122_12295

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : current = 7 → total = 16 → cost = 72 → (cost / (total - current) : ℚ) = 8 := by
  sorry

end action_figure_cost_l122_12295


namespace inequality_proof_l122_12255

/-- Proves that given a = 0.1e^0.1, b = 1/9, and c = -ln 0.9, the inequality c < a < b holds -/
theorem inequality_proof (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1) 
  (hb : b = 1 / 9) 
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
  sorry

end inequality_proof_l122_12255


namespace multiple_properties_l122_12268

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
sorry

end multiple_properties_l122_12268


namespace train_passing_jogger_l122_12296

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger
  (jogger_speed : Real)
  (train_speed : Real)
  (train_length : Real)
  (initial_distance : Real)
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
  sorry

end train_passing_jogger_l122_12296


namespace fifth_month_sale_l122_12231

def average_sale : ℕ := 5600
def num_months : ℕ := 6
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month4 : ℕ := 7200
def sale_month6 : ℕ := 1200

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 4500 := by
  sorry

end fifth_month_sale_l122_12231


namespace min_value_expression_l122_12280

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 * Real.sqrt 2 ∧
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) = 3 * Real.sqrt 2 ↔ y = x * Real.sqrt 2 :=
by sorry

end min_value_expression_l122_12280


namespace inequality_holds_iff_theta_in_range_l122_12236

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inequality_holds_iff_theta_in_range :
  ∀ x θ : ℝ,
  x ≥ 3/2 →
  0 < θ →
  θ < π →
  (f (x / Real.sin θ) - (4 * (Real.sin θ)^2 * f x) ≤ f (x - 1) + 4 * f (Real.sin θ))
  ↔
  π/3 ≤ θ ∧ θ ≤ 2*π/3 :=
by sorry

end inequality_holds_iff_theta_in_range_l122_12236


namespace prob_at_least_one_heart_or_king_l122_12242

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of cards that are either hearts or kings
def heart_or_king : ℕ := 16

-- Define the probability of not choosing a heart or king in one draw
def prob_not_heart_or_king : ℚ := (total_cards - heart_or_king) / total_cards

-- Theorem statement
theorem prob_at_least_one_heart_or_king :
  1 - prob_not_heart_or_king ^ 2 = 88 / 169 := by
  sorry

end prob_at_least_one_heart_or_king_l122_12242


namespace quadratic_inequality_solution_l122_12237

theorem quadratic_inequality_solution (b : ℝ) :
  (∀ x, x^2 - 3*x + 6 > 4 ↔ (x < 1 ∨ x > b)) →
  (b = 2 ∧
   ∀ c, 
     (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
     (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2) ∧
     (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0))) :=
by sorry

end quadratic_inequality_solution_l122_12237


namespace don_rum_limit_l122_12201

/-- The amount of rum Sally gave Don on his pancakes (in oz) -/
def sally_rum : ℝ := 10

/-- The multiplier for the maximum amount of rum Don can consume for a healthy diet -/
def max_multiplier : ℝ := 3

/-- The amount of rum Don had earlier that day (in oz) -/
def earlier_rum : ℝ := 12

/-- The amount of rum Don can have after eating all of the rum and pancakes (in oz) -/
def remaining_rum : ℝ := max_multiplier * sally_rum - earlier_rum

theorem don_rum_limit : remaining_rum = 18 := by sorry

end don_rum_limit_l122_12201


namespace f_positive_iff_a_in_range_l122_12211

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.log (1 / (a * x + a)) - a

theorem f_positive_iff_a_in_range (a : ℝ) :
  (a > 0 ∧ ∀ x, f a x > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end f_positive_iff_a_in_range_l122_12211


namespace loan_to_c_amount_lent_to_c_l122_12289

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_to_c (loan_to_b : ℝ) (time_b : ℝ) (time_c : ℝ) (total_interest : ℝ) (rate : ℝ) : ℝ :=
  let interest_b := simple_interest loan_to_b rate time_b
  let interest_c := total_interest - interest_b
  interest_c / (rate * time_c)

/-- The amount A lent to C --/
theorem amount_lent_to_c : loan_to_c 5000 2 4 2640 0.12 = 3000 := by
  sorry

end loan_to_c_amount_lent_to_c_l122_12289


namespace approximate_4_02_to_ten_thousandth_l122_12210

/-- Represents a decimal number with a specific precision -/
structure DecimalNumber where
  value : ℚ
  precision : ℕ

/-- Represents the place value in a decimal number -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths

/-- Determines the place value of the last non-zero digit in a decimal number -/
def lastNonZeroDigitPlace (n : DecimalNumber) : PlaceValue :=
  sorry

/-- Approximates a decimal number to a given place value -/
def approximateTo (n : DecimalNumber) (place : PlaceValue) : DecimalNumber :=
  sorry

/-- Theorem stating that approximating 4.02 to the ten thousandth place
    results in a number accurate to the hundredth place -/
theorem approximate_4_02_to_ten_thousandth :
  let original := DecimalNumber.mk (402 / 100) 2
  let approximated := approximateTo original PlaceValue.TenThousandths
  lastNonZeroDigitPlace approximated = PlaceValue.Hundredths :=
sorry

end approximate_4_02_to_ten_thousandth_l122_12210


namespace orange_apple_cost_l122_12290

/-- The cost of oranges and apples given specific quantities and price per kilo -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (orange_quantity apple_quantity : ℕ) : 
  orange_price = 29 → 
  apple_price = 29 → 
  orange_quantity = 6 → 
  apple_quantity = 5 → 
  orange_price * orange_quantity + apple_price * apple_quantity = 319 :=
by
  sorry

#check orange_apple_cost

end orange_apple_cost_l122_12290


namespace sue_falls_count_l122_12264

structure Friend where
  name : String
  age : Nat
  falls : Nat

def steven : Friend := { name := "Steven", age := 20, falls := 3 }
def stephanie : Friend := { name := "Stephanie", age := 24, falls := steven.falls + 13 }
def sam : Friend := { name := "Sam", age := 24, falls := 1 }
def sue : Friend := { name := "Sue", age := 26, falls := 0 }  -- falls will be calculated

def sonya_falls : Nat := stephanie.falls / 2 - 2
def sophie_falls : Nat := sam.falls + 4

def youngest_age : Nat := min steven.age (min stephanie.age (min sam.age sue.age))

theorem sue_falls_count :
  sue.falls = sue.age - youngest_age :=
by sorry

end sue_falls_count_l122_12264


namespace tonya_lemonade_revenue_l122_12205

/-- Calculates the total revenue from Tonya's lemonade stand --/
def lemonade_revenue (small_price medium_price large_price : ℕ)
  (small_revenue medium_revenue : ℕ) (large_cups : ℕ) : ℕ :=
  small_revenue + medium_revenue + (large_cups * large_price)

theorem tonya_lemonade_revenue :
  lemonade_revenue 1 2 3 11 24 5 = 50 := by
  sorry

end tonya_lemonade_revenue_l122_12205


namespace calculation_proof_l122_12208

theorem calculation_proof : 8 * 2.25 - 5 * 0.85 / 2.5 = 16.3 := by
  sorry

end calculation_proof_l122_12208


namespace fraction_equality_l122_12243

theorem fraction_equality (x z : ℚ) (hx : x = 4 / 7) (hz : z = 8 / 11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end fraction_equality_l122_12243


namespace triangle_cannot_be_formed_l122_12223

theorem triangle_cannot_be_formed (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 9) : 
  ¬ (∃ (a' b' c' : ℝ), a' = a * 1.5 ∧ b' = b * (1 - 0.333) ∧ c' = c ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a') :=
by sorry

end triangle_cannot_be_formed_l122_12223


namespace president_vice_president_selection_l122_12279

/-- The number of ways to select a president and a vice president from a group of 4 people -/
def select_president_and_vice_president (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a president and a vice president from a group of 4 people is 12 -/
theorem president_vice_president_selection :
  select_president_and_vice_president 4 = 12 := by
  sorry

end president_vice_president_selection_l122_12279


namespace geometric_progression_x_value_l122_12262

theorem geometric_progression_x_value : 
  ∀ (x : ℝ), 
  let a₁ := 2*x - 2
  let a₂ := 2*x + 2
  let a₃ := 4*x + 6
  (a₂ / a₁ = a₃ / a₂) → x = -2 :=
by
  sorry

end geometric_progression_x_value_l122_12262


namespace q_div_p_equals_48_l122_12200

/-- The number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 13

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The probability of drawing all 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 of another -/
def q : ℚ := (624 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating the ratio of q to p -/
theorem q_div_p_equals_48 : q / p = 48 := by sorry

end q_div_p_equals_48_l122_12200


namespace book_price_is_480_l122_12260

/-- The price of a book that Tara sells to reach her goal of buying a clarinet -/
def book_price : ℚ :=
  let clarinet_cost : ℚ := 90
  let initial_savings : ℚ := 10
  let needed_amount : ℚ := clarinet_cost - initial_savings
  let lost_savings : ℚ := needed_amount / 2
  let total_to_save : ℚ := needed_amount + lost_savings
  let num_books : ℚ := 25
  total_to_save / num_books

/-- Theorem stating that the book price is $4.80 -/
theorem book_price_is_480 : book_price = 4.8 := by
  sorry

end book_price_is_480_l122_12260


namespace unique_two_digit_number_l122_12202

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ x y : ℕ, n = 10 * x + y ∧ 
    10 ≤ x + y ∧ x + y < 100 ∧
    x = y / 4 ∧
    n = 28) :=
by sorry

end unique_two_digit_number_l122_12202


namespace diff_suit_prob_is_13_17_l122_12283

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- A function that assigns a suit to each card in the deck -/
def card_suit : Fin 52 → Suit := sorry

/-- The probability of picking two cards of different suits -/
def diff_suit_prob (d : Deck) : ℚ :=
  (39 : ℚ) / 51

/-- Theorem stating that the probability of picking two cards of different suits is 13/17 -/
theorem diff_suit_prob_is_13_17 (d : Deck) :
  diff_suit_prob d = 13 / 17 := by
  sorry

end diff_suit_prob_is_13_17_l122_12283


namespace least_whole_number_subtraction_l122_12299

theorem least_whole_number_subtraction (x : ℕ) : 
  x ≥ 3 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end least_whole_number_subtraction_l122_12299


namespace unique_base_ten_l122_12247

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if the equation is valid in base b -/
def isValidEquation (b : Nat) : Prop :=
  toDecimal [8, 7, 3, 6, 4] b + toDecimal [9, 2, 4, 1, 7] b = toDecimal [1, 8, 5, 8, 7, 1] b

theorem unique_base_ten :
  ∃! b, isValidEquation b ∧ b = 10 := by sorry

end unique_base_ten_l122_12247


namespace football_yards_gained_l122_12220

/-- Represents the yards gained by a football team after an initial loss -/
def yards_gained (initial_loss : ℤ) (final_progress : ℤ) : ℤ :=
  final_progress - initial_loss

/-- Theorem: If a team loses 5 yards and ends with 6 yards of progress, they gained 11 yards -/
theorem football_yards_gained :
  yards_gained (-5) 6 = 11 := by
  sorry

end football_yards_gained_l122_12220


namespace chocolate_theorem_l122_12217

/-- The difference between 75% of Robert's chocolates and the total number of chocolates Nickel and Penelope ate -/
def chocolate_difference (robert : ℝ) (nickel : ℝ) (penelope : ℝ) : ℝ :=
  0.75 * robert - (nickel + penelope)

/-- Theorem stating the difference in chocolates -/
theorem chocolate_theorem :
  chocolate_difference 13 4 7.5 = -1.75 := by
  sorry

end chocolate_theorem_l122_12217


namespace tan_graph_problem_l122_12259

theorem tan_graph_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 4) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 3 * π / 4))) →
  a * b = 4 * Real.sqrt 3 / 3 := by
  sorry

end tan_graph_problem_l122_12259


namespace fraction_evaluation_l122_12293

theorem fraction_evaluation : (3^4 - 3^2) / (3^(-2) + 3^(-4)) = 583.2 := by
  sorry

end fraction_evaluation_l122_12293


namespace fence_cost_square_plot_l122_12250

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3808 := by
sorry

end fence_cost_square_plot_l122_12250


namespace total_scheduling_arrangements_l122_12224

/-- Represents the total number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 2

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Math in the morning -/
def math_morning_options : ℕ := 4

/-- Represents the number of ways to schedule Physical Education (excluding first morning period) -/
def pe_options : ℕ := 5

/-- Represents the number of ways to arrange the remaining subjects -/
def remaining_arrangements : ℕ := 24

/-- Theorem stating the total number of different scheduling arrangements -/
theorem total_scheduling_arrangements :
  math_morning_options * pe_options * remaining_arrangements = 480 := by
  sorry

end total_scheduling_arrangements_l122_12224


namespace star_six_five_l122_12206

-- Define the star operation
def star (a b : ℕ+) : ℚ :=
  (a.val * (2 * b.val)) / (a.val + 2 * b.val + 3)

-- Theorem statement
theorem star_six_five :
  star 6 5 = 60 / 19 := by
  sorry

end star_six_five_l122_12206


namespace union_complement_equals_reals_l122_12277

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 > 2*x + 3}

noncomputable def B : Set ℝ := {x | Real.log x / Real.log 3 > 1}

theorem union_complement_equals_reals : A ∪ (U \ B) = U := by sorry

end union_complement_equals_reals_l122_12277


namespace last_digit_base_9_of_221122211111_base_3_l122_12209

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def last_digit_base_9 (n : Nat) : Nat :=
  n % 9

theorem last_digit_base_9_of_221122211111_base_3 :
  let y : Nat := base_3_to_10 [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2]
  last_digit_base_9 y = 6 := by
  sorry

end last_digit_base_9_of_221122211111_base_3_l122_12209


namespace set_inclusion_condition_l122_12244

/-- The necessary and sufficient condition for set inclusion -/
theorem set_inclusion_condition (a : ℝ) (h : a > 0) :
  ({p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 4)^2 ≤ 1} ⊆ 
   {p : ℝ × ℝ | |p.1 - 3| + 2 * |p.2 + 4| ≤ a}) ↔ 
  a ≥ Real.sqrt 5 :=
by sorry

end set_inclusion_condition_l122_12244


namespace balls_sold_prove_balls_sold_l122_12288

/-- The number of balls sold given the cost price, selling price, and loss condition. -/
theorem balls_sold (cost_price : ℕ) (selling_price : ℕ) (loss : ℕ) : ℕ :=
  let n : ℕ := (selling_price + loss) / cost_price
  n

/-- Prove that 13 balls were sold given the problem conditions. -/
theorem prove_balls_sold :
  balls_sold 90 720 (5 * 90) = 13 := by
  sorry

end balls_sold_prove_balls_sold_l122_12288


namespace not_sufficient_not_necessary_l122_12294

theorem not_sufficient_not_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, (a < 0 ∧ b < 0) → a * b * (a - b) > 0) ∧ 
  ¬(∀ a b : ℝ, a * b * (a - b) > 0 → (a < 0 ∧ b < 0)) := by
  sorry

end not_sufficient_not_necessary_l122_12294


namespace triangle_area_proof_l122_12252

/-- The area of the triangle formed by y = x, x = -5, and the x-axis --/
def triangle_area : ℝ := 12.5

/-- The x-coordinate of the vertical line --/
def vertical_line_x : ℝ := -5

/-- Theorem: The area of the triangle formed by y = x, x = -5, and the x-axis is 12.5 --/
theorem triangle_area_proof :
  let intersection_point := (vertical_line_x, vertical_line_x)
  let base := -vertical_line_x
  let height := -vertical_line_x
  (1/2 : ℝ) * base * height = triangle_area := by
  sorry

end triangle_area_proof_l122_12252


namespace square_of_9_divided_by_cube_root_of_125_remainder_l122_12235

theorem square_of_9_divided_by_cube_root_of_125_remainder (n m q r : ℕ) : 
  n = 9^2 → 
  m = 5 → 
  n = m * q + r → 
  r < m → 
  r = 1 :=
by
  sorry

end square_of_9_divided_by_cube_root_of_125_remainder_l122_12235


namespace boys_who_watched_l122_12240

/-- The number of boys who went down the slide initially -/
def x : ℕ := 22

/-- The number of additional boys who went down the slide later -/
def y : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_slide : ℕ := x + y

/-- The ratio of boys who went down the slide to boys who watched -/
def ratio_slide_to_watch : Rat := 5 / 3

/-- The number of boys who watched but didn't go down the slide -/
def z : ℕ := (3 * total_slide) / 5

theorem boys_who_watched (h : ratio_slide_to_watch = 5 / 3) : z = 21 := by
  sorry

end boys_who_watched_l122_12240


namespace f_values_l122_12225

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else if x ≥ 0 then Real.exp (x - 1)
  else 0  -- undefined for x ≤ -1

theorem f_values (a : ℝ) : f 1 + f a = 2 → a = 1 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end f_values_l122_12225


namespace denise_crayon_sharing_l122_12253

/-- The number of crayons Denise has -/
def total_crayons : ℕ := 210

/-- The number of crayons each friend gets -/
def crayons_per_friend : ℕ := 7

/-- The number of friends Denise shares crayons with -/
def number_of_friends : ℕ := total_crayons / crayons_per_friend

theorem denise_crayon_sharing :
  number_of_friends = 30 := by sorry

end denise_crayon_sharing_l122_12253


namespace derivative_of_fraction_l122_12245

open Real

theorem derivative_of_fraction (x : ℝ) (h : x > 0) :
  deriv (λ x => (1 - log x) / (1 + log x)) x = -2 / (x * (1 + log x)^2) := by
  sorry

end derivative_of_fraction_l122_12245


namespace segment_length_specific_case_l122_12267

/-- A rectangle with an inscribed circle and a diagonal intersecting the circle -/
structure RectangleWithCircle where
  /-- Length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- Length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  circle_tangent : Bool
  /-- The diagonal intersects the circle at two points -/
  diagonal_intersects : Bool

/-- The length of the segment AB formed by the intersection of the diagonal with the circle -/
def segment_length (r : RectangleWithCircle) : ℝ :=
  sorry

/-- Theorem stating the length of AB in the specific case -/
theorem segment_length_specific_case :
  let r : RectangleWithCircle := {
    short_side := 2,
    long_side := 4,
    circle_tangent := true,
    diagonal_intersects := true
  }
  segment_length r = 4 * Real.sqrt 5 / 5 :=
sorry

end segment_length_specific_case_l122_12267


namespace jenny_reading_time_l122_12271

/-- Calculates the average daily reading time including breaks -/
def averageDailyReadingTime (numBooks : ℕ) (totalDays : ℕ) (readingSpeed : ℕ) 
  (breakDuration : ℕ) (breakInterval : ℕ) (bookWords : List ℕ) : ℕ :=
  let totalWords := bookWords.sum
  let readingMinutes := totalWords / readingSpeed
  let readingHours := readingMinutes / 60
  let numBreaks := readingHours
  let breakMinutes := numBreaks * breakDuration
  let totalMinutes := readingMinutes + breakMinutes
  totalMinutes / totalDays

/-- Theorem: Jenny's average daily reading time is 124 minutes -/
theorem jenny_reading_time :
  let numBooks := 5
  let totalDays := 15
  let readingSpeed := 60  -- words per minute
  let breakDuration := 15  -- minutes
  let breakInterval := 60  -- minutes
  let bookWords := [12000, 18000, 24000, 15000, 21000]
  averageDailyReadingTime numBooks totalDays readingSpeed breakDuration breakInterval bookWords = 124 := by
  sorry

end jenny_reading_time_l122_12271


namespace carpet_area_and_cost_exceed_limits_l122_12229

/-- Represents the dimensions of various room types in Jesse's house -/
structure RoomDimensions where
  rectangular_length : ℝ
  rectangular_width : ℝ
  square_side : ℝ
  triangular_base : ℝ
  triangular_height : ℝ
  trapezoidal_base1 : ℝ
  trapezoidal_base2 : ℝ
  trapezoidal_height : ℝ
  circular_radius : ℝ
  elliptical_major_axis : ℝ
  elliptical_minor_axis : ℝ

/-- Represents the number of each room type in Jesse's house -/
structure RoomCounts where
  rectangular : ℕ
  square : ℕ
  triangular : ℕ
  trapezoidal : ℕ
  circular : ℕ
  elliptical : ℕ

/-- Calculates the total carpet area needed and proves it exceeds 2000 square feet -/
def total_carpet_area_exceeds_2000 (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area > 2000

/-- Proves that the total cost exceeds $10,000 when carpet costs $5 per square foot -/
def total_cost_exceeds_budget (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area * 5 > 10000

/-- Main theorem combining both conditions -/
theorem carpet_area_and_cost_exceed_limits (dims : RoomDimensions) (counts : RoomCounts) :
  total_carpet_area_exceeds_2000 dims counts ∧ total_cost_exceeds_budget dims counts :=
sorry

end carpet_area_and_cost_exceed_limits_l122_12229


namespace reflection_sum_l122_12234

theorem reflection_sum (x : ℝ) : 
  let C : ℝ × ℝ := (x, -3)
  let D : ℝ × ℝ := (-x, -3)
  (C.1 + C.2 + D.1 + D.2) = -6 := by sorry

end reflection_sum_l122_12234


namespace range_of_a_l122_12204

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 33 > 0
def q (x a : ℝ) : Prop := |x - 1| > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, a > 0 → (p x → q x a) ∧ ¬(q x a → p x)) :
  ∃ a : ℝ, a > 0 ∧ a ≤ 4 ∧ ∀ b : ℝ, (b > 0 ∧ b ≤ 4 → ∃ x : ℝ, p x → q x b) ∧
    (b > 4 → ∃ x : ℝ, p x ∧ ¬(q x b)) :=
sorry

end range_of_a_l122_12204


namespace pumpkin_difference_l122_12238

theorem pumpkin_difference (moonglow_pumpkins sunshine_pumpkins : ℕ) 
  (h1 : moonglow_pumpkins = 14)
  (h2 : sunshine_pumpkins = 54) :
  sunshine_pumpkins - 3 * moonglow_pumpkins = 12 := by
  sorry

end pumpkin_difference_l122_12238


namespace slope_of_line_l122_12221

/-- The slope of the line (x/4) + (y/5) = 1 is -5/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) := by
sorry

end slope_of_line_l122_12221


namespace second_car_speed_l122_12285

/-- Two cars traveling on a road in the same direction -/
structure TwoCars where
  /-- Time of travel in seconds -/
  t : ℝ
  /-- Average speed of the first car in m/s -/
  v₁ : ℝ
  /-- Initial distance between cars in meters -/
  S₁ : ℝ
  /-- Final distance between cars in meters -/
  S₂ : ℝ

/-- Average speed of the second car -/
def averageSpeedSecondCar (cars : TwoCars) : Set ℝ :=
  let v_rel := (cars.S₁ - cars.S₂) / cars.t
  {cars.v₁ - v_rel, cars.v₁ + v_rel}

/-- Theorem stating the average speed of the second car -/
theorem second_car_speed (cars : TwoCars)
    (h_t : cars.t = 30)
    (h_v₁ : cars.v₁ = 30)
    (h_S₁ : cars.S₁ = 800)
    (h_S₂ : cars.S₂ = 200) :
    averageSpeedSecondCar cars = {10, 50} := by
  sorry

end second_car_speed_l122_12285


namespace symmetric_points_difference_l122_12275

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The main theorem stating that if A(a,-2) and B(4,b) are symmetric with respect to the origin, then a-b = -6 -/
theorem symmetric_points_difference (a b : ℝ) : 
  symmetric_wrt_origin a (-2) 4 b → a - b = -6 := by
  sorry

end symmetric_points_difference_l122_12275


namespace f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l122_12273

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

-- Statement 1: f is odd when a = 0
theorem f_odd_when_a_zero : 
  ∀ x : ℝ, f 0 (-x) = -(f 0 x) :=
sorry

-- Statement 2: f is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Statement 3: f(x) - tf(2a) = 0 has three distinct roots iff 1 < t < 9/8
theorem three_roots_iff :
  ∀ a t : ℝ, a ∈ Set.Icc (-2) 2 →
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x - t * f a (2 * a) = 0 ∧
    f a y - t * f a (2 * a) = 0 ∧
    f a z - t * f a (2 * a) = 0) ↔
  1 < t ∧ t < 9/8 :=
sorry

end f_odd_when_a_zero_f_increasing_iff_three_roots_iff_l122_12273


namespace complement_union_eq_five_l122_12281

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_eq_five : (U \ (M ∪ N)) = {5} := by sorry

end complement_union_eq_five_l122_12281


namespace intersection_of_A_and_B_l122_12251

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = 1 / x}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≠ 0} := by sorry

end intersection_of_A_and_B_l122_12251


namespace max_value_expression_l122_12292

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0) 
  (sum_condition : x + y + z = 2) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 256/243 := by
  sorry

end max_value_expression_l122_12292


namespace opposite_face_is_D_l122_12203

-- Define a cube net
structure CubeNet :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)

-- Define a cube
structure Cube :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)
  (opposite : Char → Char)
  (opposite_symm : ∀ x, opposite (opposite x) = x)

-- Define the folding operation
def fold (net : CubeNet) : Cube :=
  { faces := net.faces,
    is_valid := net.is_valid,
    opposite := sorry,
    opposite_symm := sorry }

-- Theorem statement
theorem opposite_face_is_D (net : CubeNet) 
  (h1 : net.faces = {'A', 'B', 'C', 'D', 'E', 'F'}) :
  (fold net).opposite 'A' = 'D' :=
sorry

end opposite_face_is_D_l122_12203


namespace elimination_theorem_l122_12298

theorem elimination_theorem (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x^3 + y^3) 
  (hc : c = x^5 + y^5) : 
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) := by
  sorry

end elimination_theorem_l122_12298


namespace d_share_is_thirteen_sixtieths_l122_12256

/-- Represents the capital shares of partners in a business. -/
structure CapitalShares where
  total : ℚ
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  a_share : a = (1 : ℚ) / 3 * total
  b_share : b = (1 : ℚ) / 4 * total
  c_share : c = (1 : ℚ) / 5 * total
  total_sum : a + b + c + d = total

/-- Represents the profit distribution in the business. -/
structure ProfitDistribution where
  total : ℚ
  a_profit : ℚ
  total_amount : total = 2490
  a_amount : a_profit = 830

/-- Theorem stating that given the capital shares and profit distribution,
    partner D's share of the capital is 13/60. -/
theorem d_share_is_thirteen_sixtieths
  (shares : CapitalShares) (profit : ProfitDistribution) :
  shares.d = (13 : ℚ) / 60 * shares.total :=
sorry

end d_share_is_thirteen_sixtieths_l122_12256


namespace root_relation_implies_a_value_l122_12270

theorem root_relation_implies_a_value (m : ℝ) (h : m > 0) :
  ∃ (a : ℝ), ∀ (x : ℂ),
    (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a →
    (∃ (y : ℂ), (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a ∧ x = m * y) →
    a = (m + 1) / (2 * m) * Real.sqrt m :=
by sorry

end root_relation_implies_a_value_l122_12270
