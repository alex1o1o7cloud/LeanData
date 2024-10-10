import Mathlib

namespace new_year_cards_profit_l1162_116293

/-- The profit calculation for a store selling New Year cards -/
theorem new_year_cards_profit
  (purchase_price : ℕ)
  (total_sale : ℕ)
  (h1 : purchase_price = 21)
  (h2 : total_sale = 1457)
  (h3 : ∃ (n : ℕ) (selling_price : ℕ), n * selling_price = total_sale ∧ selling_price ≤ 2 * purchase_price) :
  ∃ (n : ℕ) (selling_price : ℕ), 
    n * selling_price = total_sale ∧ 
    selling_price ≤ 2 * purchase_price ∧
    n * (selling_price - purchase_price) = 470 :=
by sorry


end new_year_cards_profit_l1162_116293


namespace bisected_line_segment_l1162_116270

/-- Given a line segment with endpoints (5,1) and (m,1) bisected by x-2y=0, m = -1 -/
theorem bisected_line_segment (m : ℝ) : 
  let endpoint1 : ℝ × ℝ := (5, 1)
  let endpoint2 : ℝ × ℝ := (m, 1)
  let bisector : ℝ → ℝ := fun x => x / 2
  (bisector (endpoint1.1 + endpoint2.1) - 2 * 1 = 0) → m = -1 := by
sorry

end bisected_line_segment_l1162_116270


namespace opposite_of_negative_three_l1162_116215

theorem opposite_of_negative_three :
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by sorry

end opposite_of_negative_three_l1162_116215


namespace complex_exponentiation_l1162_116299

theorem complex_exponentiation (i : ℂ) (h : i * i = -1) : 
  (1 + i) ^ (2 * i) = -2 := by
  sorry

end complex_exponentiation_l1162_116299


namespace correct_operation_is_multiplication_by_three_l1162_116261

theorem correct_operation_is_multiplication_by_three (x : ℝ) : 
  (((3 * x - x / 5) / (3 * x)) * 100 = 93.33333333333333) → 
  (∃ (y : ℝ), y = 3 ∧ x * y = 3 * x) :=
by
  sorry

end correct_operation_is_multiplication_by_three_l1162_116261


namespace expression_simplification_l1162_116218

theorem expression_simplification (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a - b = 2) : 
  (a^2 - 6*a*b + 9*b^2) / (a^2 - 2*a*b) / 
  ((5*b^2) / (a - 2*b) - a - 2*b) - 1/a = -1/3 :=
by sorry

end expression_simplification_l1162_116218


namespace rectangular_field_area_l1162_116295

/-- Represents a rectangular field with a specific ratio of width to length and a given perimeter. -/
structure RectangularField where
  width : ℝ
  length : ℝ
  width_length_ratio : width = length / 3
  perimeter : width * 2 + length * 2 = 72

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that a rectangular field with the given properties has an area of 243 square meters. -/
theorem rectangular_field_area (field : RectangularField) : area field = 243 := by
  sorry

end rectangular_field_area_l1162_116295


namespace tom_apple_problem_l1162_116273

theorem tom_apple_problem (num_apples : ℕ) : 
  let total_slices := num_apples * 8
  let remaining_after_jerry := total_slices * (5/8 : ℚ)
  let remaining_after_eating := remaining_after_jerry * (1/2 : ℚ)
  remaining_after_eating = 5 →
  num_apples = 2 := by
sorry

end tom_apple_problem_l1162_116273


namespace garden_area_is_855_l1162_116232

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  posts : ℕ
  post_distance : ℝ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given the specifications -/
def garden_area (g : Garden) : ℝ :=
  let shorter_side_posts := (g.posts / 2) / (g.longer_side_post_ratio + 1)
  let longer_side_posts := g.longer_side_post_ratio * shorter_side_posts
  let shorter_side_length := (shorter_side_posts - 1) * g.post_distance
  let longer_side_length := (longer_side_posts - 1) * g.post_distance
  shorter_side_length * longer_side_length

/-- Theorem stating that the garden with given specifications has an area of 855 square yards -/
theorem garden_area_is_855 (g : Garden) 
    (h1 : g.posts = 24)
    (h2 : g.post_distance = 6)
    (h3 : g.longer_side_post_ratio = 3) : 
  garden_area g = 855 := by
  sorry

end garden_area_is_855_l1162_116232


namespace banana_mango_equivalence_l1162_116229

/-- Represents the cost relationship between fruits -/
structure FruitCost where
  banana : ℝ
  pear : ℝ
  mango : ℝ

/-- The given cost relationships -/
def cost_relation (c : FruitCost) : Prop :=
  4 * c.banana = 3 * c.pear ∧ 8 * c.pear = 5 * c.mango

/-- The theorem to prove -/
theorem banana_mango_equivalence (c : FruitCost) (h : cost_relation c) :
  20 * c.banana = 9.375 * c.mango :=
sorry

end banana_mango_equivalence_l1162_116229


namespace representable_multiple_of_three_l1162_116269

/-- A number is representable if it can be written as x^2 + 2y^2 for some integers x and y -/
def Representable (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*y^2

/-- If 3a is representable, then a is representable -/
theorem representable_multiple_of_three (a : ℤ) :
  Representable (3*a) → Representable a := by
  sorry

end representable_multiple_of_three_l1162_116269


namespace perpendicular_vectors_l1162_116200

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a - b), then the x-coordinate of b is 1/2. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b.2 = 4) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.1 = 1/2 := by
  sorry

end perpendicular_vectors_l1162_116200


namespace negation_of_exists_prop_l1162_116241

theorem negation_of_exists_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_exists_prop_l1162_116241


namespace triangle_theorem_l1162_116245

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Triangle ABC is acute
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  -- Sum of angles is π
  A + B + C = Real.pi ∧
  -- Given conditions
  a = 2*b * Real.sin A ∧
  a = 3 * Real.sqrt 3 ∧
  c = 5 →
  -- Conclusions
  B = Real.pi/6 ∧  -- 30° in radians
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4) ∧
  b = Real.sqrt 7

theorem triangle_theorem :
  ∀ a b c A B C, triangle_proof a b c A B C :=
sorry

end triangle_theorem_l1162_116245


namespace unique_prime_triple_l1162_116244

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  Nat.Prime (4 * q - 1) ∧
  (p + q : ℚ) / (p + r) = r - p ∧
  p = 2 ∧ q = 3 ∧ r = 3 := by
sorry

end unique_prime_triple_l1162_116244


namespace evaluate_expression_l1162_116214

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4^2 * 3) = -24 := by
  sorry

end evaluate_expression_l1162_116214


namespace average_salary_non_technicians_l1162_116290

/-- Proves that the average salary of non-technician workers is 6000 given the conditions --/
theorem average_salary_non_technicians (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_technicians) / 
     (total_workers - num_technicians)) = 6000 * (total_workers - num_technicians) :=
by
  sorry

#check average_salary_non_technicians

end average_salary_non_technicians_l1162_116290


namespace merry_and_brother_lambs_l1162_116258

/-- The number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  let merry_lambs : ℕ := 10
  let brother_lambs : ℕ := merry_lambs + 3
  merry_lambs + brother_lambs = 23 :=
by sorry

end merry_and_brother_lambs_l1162_116258


namespace cheryl_pesto_production_l1162_116294

/-- The number of cups of basil needed to make one cup of pesto -/
def basil_per_pesto : ℕ := 4

/-- The number of cups of basil Cheryl can harvest per week -/
def basil_per_week : ℕ := 16

/-- The number of weeks Cheryl can harvest basil -/
def harvest_weeks : ℕ := 8

/-- The total number of cups of pesto Cheryl can make -/
def total_pesto : ℕ := (basil_per_week * harvest_weeks) / basil_per_pesto

theorem cheryl_pesto_production :
  total_pesto = 32 := by sorry

end cheryl_pesto_production_l1162_116294


namespace remainder_theorem_l1162_116286

theorem remainder_theorem (x y z : ℤ) 
  (hx : x % 102 = 56)
  (hy : y % 154 = 79)
  (hz : z % 297 = 183) :
  (x % 19 = 18) ∧ (y % 22 = 13) ∧ (z % 33 = 18) := by
  sorry

end remainder_theorem_l1162_116286


namespace pushup_problem_l1162_116262

theorem pushup_problem (x : ℕ) (h : x = 51) : 
  let zachary := x
  let melanie := 2 * zachary - 7
  let david := zachary + 22
  let karen := (zachary + melanie + david) / 3 - 5
  let john := david - 4
  john + melanie + karen = 232 := by
sorry

end pushup_problem_l1162_116262


namespace arithmetic_mean_of_neg_one_and_five_l1162_116217

theorem arithmetic_mean_of_neg_one_and_five (x y : ℝ) : 
  x = -1 → y = 5 → (x + y) / 2 = 2 := by
  sorry

end arithmetic_mean_of_neg_one_and_five_l1162_116217


namespace album_time_calculation_l1162_116233

/-- Calculates the total time to finish all songs in an album -/
def total_album_time (initial_songs : ℕ) (song_duration : ℕ) (added_songs : ℕ) : ℕ :=
  (initial_songs + added_songs) * song_duration

/-- Theorem: Given an initial album of 25 songs, each 3 minutes long, and adding 10 more songs
    of the same duration, the total time to finish all songs in the album is 105 minutes. -/
theorem album_time_calculation :
  total_album_time 25 3 10 = 105 := by
  sorry

end album_time_calculation_l1162_116233


namespace g_minus_g_is_zero_l1162_116253

def f : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 2 = 0 then 2 * f (n / 2) + 1 else 2 * f n

def g (n : ℕ) : ℕ := f (f n)

theorem g_minus_g_is_zero (n : ℕ) : g (n - g n) = 0 := by
  sorry

end g_minus_g_is_zero_l1162_116253


namespace magic_square_sum_l1162_116289

/-- Represents a 3x3 magic square -/
def MagicSquare := Fin 3 → Fin 3 → ℕ

/-- The magic sum of a magic square -/
def magicSum (s : MagicSquare) : ℕ := s 0 0 + s 0 1 + s 0 2

/-- Predicate to check if a square is magic -/
def isMagic (s : MagicSquare) : Prop :=
  let sum := magicSum s
  (∀ i, s i 0 + s i 1 + s i 2 = sum) ∧
  (∀ j, s 0 j + s 1 j + s 2 j = sum) ∧
  (s 0 0 + s 1 1 + s 2 2 = sum) ∧
  (s 0 2 + s 1 1 + s 2 0 = sum)

theorem magic_square_sum (s : MagicSquare) (x y : ℕ) 
  (h1 : s 0 0 = x)
  (h2 : s 0 1 = 6)
  (h3 : s 0 2 = 20)
  (h4 : s 1 0 = 22)
  (h5 : s 1 1 = y)
  (h6 : isMagic s) :
  x + y = 12 := by
  sorry


end magic_square_sum_l1162_116289


namespace cans_for_final_rooms_l1162_116265

-- Define the initial and final number of rooms that can be painted
def initial_rooms : ℕ := 50
def final_rooms : ℕ := 42

-- Define the number of cans lost
def cans_lost : ℕ := 4

-- Define the function to calculate the number of cans needed for a given number of rooms
def cans_needed (rooms : ℕ) : ℕ :=
  rooms / ((initial_rooms - final_rooms) / cans_lost)

-- Theorem statement
theorem cans_for_final_rooms :
  cans_needed final_rooms = 21 :=
sorry

end cans_for_final_rooms_l1162_116265


namespace trig_identity_l1162_116210

theorem trig_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end trig_identity_l1162_116210


namespace fraction_equality_l1162_116296

theorem fraction_equality (n : ℝ) (h : n ≥ 2) :
  1 / (n^2 - 1) = (1/2) * (1 / (n - 1) - 1 / (n + 1)) := by
  sorry

end fraction_equality_l1162_116296


namespace sqrt_five_power_l1162_116249

theorem sqrt_five_power : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end sqrt_five_power_l1162_116249


namespace dance_class_girls_l1162_116267

theorem dance_class_girls (total : ℕ) (g b : ℚ) : 
  total = 28 →
  g / b = 3 / 4 →
  g + b = total →
  g = 12 := by sorry

end dance_class_girls_l1162_116267


namespace log_equation_solution_l1162_116252

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 - 3 * (Real.log x / Real.log 2) = 6 →
  x = (1 : ℝ) / 2^(9/4) :=
by sorry

end log_equation_solution_l1162_116252


namespace winning_candidate_vote_percentage_l1162_116274

theorem winning_candidate_vote_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (winning_percentage : ℚ) 
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_percentage = 60 / 100) : 
  (((votes_cast : ℚ) * winning_percentage) / (total_members : ℚ)) * 100 = 19.6875 := by
  sorry

end winning_candidate_vote_percentage_l1162_116274


namespace not_necessarily_right_triangle_l1162_116225

/-- A triangle with side lengths proportional to 3:4:6 is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a / b = 3 / 4 ∧ b / c = 4 / 6) :
  ¬ (a^2 + b^2 = c^2) :=
sorry

end not_necessarily_right_triangle_l1162_116225


namespace ellipse_and_dot_product_range_l1162_116279

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

-- Define the dot product of OA and OB
def dot_product (xa ya xb yb : ℝ) : ℝ := xa * xb + ya * yb

theorem ellipse_and_dot_product_range :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 →
  (∀ x y, ellipse a b x y → x^2 / a^2 + y^2 / b^2 = 1) →
  a^2 / b^2 - 1 = 1/4 →
  (∃ x, hyperbola 0 x) →
  (∀ m : ℝ, m ≠ 0 → ∃ xa ya xb yb,
    line m xa ya ∧ line m xb yb ∧
    ellipse a b xa ya ∧ ellipse a b xb yb) →
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ xa ya xb yb,
    ellipse a b xa ya ∧ ellipse a b xb yb →
    -4 ≤ dot_product xa ya xb yb ∧ dot_product xa ya xb yb < 13/4) :=
by sorry

end ellipse_and_dot_product_range_l1162_116279


namespace fermat_min_l1162_116266

theorem fermat_min (n : ℕ) (x y z : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n := by
  sorry

end fermat_min_l1162_116266


namespace problem_solution_l1162_116285

theorem problem_solution :
  (∀ x : ℝ, x + 1/x = 5 → x^2 + 1/x^2 = 23) ∧
  ((5/3)^2004 * (3/5)^2003 = 5/3) := by
sorry

end problem_solution_l1162_116285


namespace heart_ten_spade_probability_l1162_116275

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of tens in a standard deck -/
def NumTens : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a specific sequence of cards -/
def SequenceProbability (firstCardProb : ℚ) (secondCardProb : ℚ) (thirdCardProb : ℚ) : ℚ :=
  firstCardProb * secondCardProb * thirdCardProb

theorem heart_ten_spade_probability :
  let probHeartNotTen := (NumHearts - 1) / StandardDeck
  let probTenAfterHeart := NumTens / (StandardDeck - 1)
  let probSpadeAfterHeartTen := NumSpades / (StandardDeck - 2)
  let probHeartTen := 1 / StandardDeck
  let probOtherTenAfterHeartTen := (NumTens - 1) / (StandardDeck - 1)
  
  SequenceProbability probHeartNotTen probTenAfterHeart probSpadeAfterHeartTen +
  SequenceProbability probHeartTen probOtherTenAfterHeartTen probSpadeAfterHeartTen = 63 / 107800 :=
by
  sorry

end heart_ten_spade_probability_l1162_116275


namespace no_simultaneous_age_ratio_l1162_116201

theorem no_simultaneous_age_ratio : ¬∃ (x : ℝ), x ≥ 0 ∧ 
  (85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)) := by
  sorry

end no_simultaneous_age_ratio_l1162_116201


namespace unusual_bicycle_spokes_l1162_116247

/-- A bicycle with an unusual spoke configuration. -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes on a bicycle. -/
def total_spokes (b : Bicycle) : ℕ := b.front_spokes + b.back_spokes

/-- Theorem: The total number of spokes on the unusual bicycle is 60. -/
theorem unusual_bicycle_spokes :
  ∃ (b : Bicycle), b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes ∧ total_spokes b = 60 :=
by
  sorry

end unusual_bicycle_spokes_l1162_116247


namespace smallest_five_digit_palindrome_divisible_by_three_l1162_116234

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The smallest five-digit palindrome divisible by 3 -/
def smallest_palindrome : ℕ := 10001

theorem smallest_five_digit_palindrome_divisible_by_three :
  is_five_digit_palindrome smallest_palindrome ∧ 
  smallest_palindrome % 3 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n → n % 3 = 0 → n ≥ smallest_palindrome := by
  sorry

#eval smallest_palindrome

end smallest_five_digit_palindrome_divisible_by_three_l1162_116234


namespace sin_cos_range_l1162_116212

theorem sin_cos_range (x y : ℝ) (h : 2 * (Real.sin x)^2 + (Real.cos y)^2 = 1) :
  ∃ (z : ℝ), (Real.sin x)^2 + (Real.cos y)^2 = z ∧ 1/2 ≤ z ∧ z ≤ 1 :=
by sorry

end sin_cos_range_l1162_116212


namespace expand_expression_l1162_116202

theorem expand_expression (x : ℝ) : 3 * (x - 7) * (x + 10) + 5 * x = 3 * x^2 + 14 * x - 210 := by
  sorry

end expand_expression_l1162_116202


namespace smallest_n_congruence_l1162_116264

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (531 * m) % 24 = (1067 * m) % 24 → m ≥ n) ∧
  (531 * n) % 24 = (1067 * n) % 24 ∧ n = 3 := by
  sorry

end smallest_n_congruence_l1162_116264


namespace magic_square_y_value_l1162_116287

/-- Represents a 3x3 modified magic square -/
structure ModifiedMagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℕ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = 
     entries 0 j + entries 1 j + entries 2 j) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = 
     entries 0 2 + entries 1 1 + entries 2 0)

/-- The theorem stating that y must be 245 in the given modified magic square -/
theorem magic_square_y_value (square : ModifiedMagicSquare) 
  (h1 : square.entries 0 1 = 25)
  (h2 : square.entries 0 2 = 120)
  (h3 : square.entries 1 0 = 5) :
  square.entries 0 0 = 245 := by
  sorry

end magic_square_y_value_l1162_116287


namespace systematic_sampling_l1162_116250

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (groups : ℕ) 
  (interval : ℕ) 
  (group_15_num : ℕ) 
  (h1 : total_students = 160) 
  (h2 : sample_size = 20) 
  (h3 : groups = 20) 
  (h4 : interval = 8) 
  (h5 : group_15_num = 116) :
  ∃ (first_group_num : ℕ), 
    first_group_num + interval * (15 - 1) = group_15_num ∧ 
    first_group_num = 4 :=
by sorry

end systematic_sampling_l1162_116250


namespace alcohol_mixture_proof_l1162_116278

/-- Proves that mixing 300 mL of 10% alcohol solution with 450 mL of 30% alcohol solution results in a 22% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 450
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.22
  
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l1162_116278


namespace max_y_over_x_for_complex_number_l1162_116236

theorem max_y_over_x_for_complex_number (x y : ℝ) :
  let z : ℂ := (x - 2) + y * I
  (Complex.abs z)^2 = 3 →
  ∃ (k : ℝ), k^2 = 3 ∧ ∀ (t : ℝ), (y / x)^2 ≤ k^2 :=
by sorry

end max_y_over_x_for_complex_number_l1162_116236


namespace base_neg_two_2019_has_six_nonzero_digits_l1162_116211

/-- Represents a number in base -2 as a list of binary digits -/
def BaseNegTwo := List Bool

/-- Converts a natural number to its base -2 representation -/
def toBaseNegTwo (n : ℕ) : BaseNegTwo :=
  sorry

/-- Counts the number of non-zero digits in a base -2 representation -/
def countNonZeroDigits (b : BaseNegTwo) : ℕ :=
  sorry

/-- Theorem: 2019 in base -2 has exactly 6 non-zero digits -/
theorem base_neg_two_2019_has_six_nonzero_digits :
  countNonZeroDigits (toBaseNegTwo 2019) = 6 := by
  sorry

end base_neg_two_2019_has_six_nonzero_digits_l1162_116211


namespace problem_statement_l1162_116257

theorem problem_statement (p q r : ℝ) 
  (h1 : p * q / (p + r) + q * r / (q + p) + r * p / (r + q) = -7)
  (h2 : p * r / (p + r) + q * p / (q + p) + r * q / (r + q) = 8) :
  q / (p + q) + r / (q + r) + p / (r + p) = 9 := by
  sorry

end problem_statement_l1162_116257


namespace line_equation_through_point_with_slope_angle_l1162_116207

/-- The equation of a line passing through (2, 3) with a slope angle of 135° -/
theorem line_equation_through_point_with_slope_angle (x y : ℝ) :
  (x + y - 5 = 0) ↔ 
  (∃ (m : ℝ), m = Real.tan (135 * π / 180) ∧ y - 3 = m * (x - 2)) := by
  sorry

end line_equation_through_point_with_slope_angle_l1162_116207


namespace otimes_two_one_l1162_116206

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - b

-- Theorem statement
theorem otimes_two_one : otimes 2 1 = 3 := by
  sorry

end otimes_two_one_l1162_116206


namespace installation_problem_l1162_116221

theorem installation_problem (x₁ x₂ x₃ k : ℕ) :
  x₁ + x₂ + x₃ ≤ 200 ∧
  x₂ = 4 * x₁ ∧
  x₃ = k * x₁ ∧
  5 * x₃ = x₂ + 99 →
  x₁ = 9 ∧ x₂ = 36 ∧ x₃ = 27 := by
sorry

end installation_problem_l1162_116221


namespace equal_probabilities_l1162_116282

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  ({red := 100, green := 0}, {red := 0, green := 100})

/-- The number of balls transferred between boxes -/
def transfer_count : ℕ := 8

/-- The final state after transferring balls -/
def final_state : Box × Box :=
  let (red_box, green_box) := initial_state
  let red_box' := {red := red_box.red - transfer_count, green := transfer_count}
  let green_box' := {red := transfer_count, green := green_box.green}
  (red_box', green_box')

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry


end equal_probabilities_l1162_116282


namespace triangle_inequality_constant_l1162_116231

theorem triangle_inequality_constant (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2) / c^2 > 1/2 ∧ ∀ N : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' > c' → b' + c' > a' → c' + a' > b' → (a'^2 + b'^2) / c'^2 > N) → N ≤ 1/2 :=
by sorry

end triangle_inequality_constant_l1162_116231


namespace slope_intercept_sum_l1162_116238

/-- Given two points C and D on a Cartesian plane, this theorem proves that
    the sum of the slope and y-intercept of the line passing through these points is 1. -/
theorem slope_intercept_sum (C D : ℝ × ℝ) : C = (2, 3) → D = (5, 9) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 1 := by
  sorry

end slope_intercept_sum_l1162_116238


namespace ellipse_area_theorem_l1162_116235

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- PF1 is perpendicular to PF2 -/
def PF1_perp_PF2 : Prop := sorry

/-- The area of triangle F1PF2 -/
def area_F1PF2 : ℝ := sorry

theorem ellipse_area_theorem :
  ellipse_equation P.1 P.2 →
  PF1_perp_PF2 →
  area_F1PF2 = 9 := by sorry

end ellipse_area_theorem_l1162_116235


namespace volume_relationship_l1162_116263

/-- Given a right circular cone, cylinder, and sphere with specific properties, 
    prove the relationship between their volumes. -/
theorem volume_relationship (h r : ℝ) (A M C : ℝ) : 
  h > 0 → r > 0 →
  A = (1/3) * π * r^2 * h →
  M = π * r^2 * (2*h) →
  C = (4/3) * π * h^3 →
  A + M - C = π * h^3 := by
  sorry


end volume_relationship_l1162_116263


namespace pauls_money_duration_l1162_116298

/-- Given Paul's earnings and weekly spending, prove how long his money will last. -/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) :
  lawn_mowing = 68 →
  weed_eating = 13 →
  weekly_spending = 9 →
  (lawn_mowing + weed_eating) / weekly_spending = 9 := by
  sorry

#check pauls_money_duration

end pauls_money_duration_l1162_116298


namespace return_flight_time_l1162_116246

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane's speed in still air
  w : ℝ  -- wind speed
  against_wind_time : ℝ -- time for flight against wind
  still_air_time : ℝ  -- time for flight in still air

/-- The conditions of the flight scenario -/
def flight_conditions (scenario : FlightScenario) : Prop :=
  scenario.against_wind_time = 120 ∧
  scenario.d = scenario.against_wind_time * (scenario.p - scenario.w) ∧
  scenario.d / (scenario.p + scenario.w) = scenario.still_air_time - 10

/-- The theorem stating that under the given conditions, the return flight time is 110 minutes -/
theorem return_flight_time (scenario : FlightScenario) 
  (h : flight_conditions scenario) : 
  scenario.d / (scenario.p + scenario.w) = 110 := by
  sorry


end return_flight_time_l1162_116246


namespace intersection_A_complement_B_l1162_116209

open Set Real

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2 ≥ 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < sqrt 2} := by sorry

end intersection_A_complement_B_l1162_116209


namespace students_without_A_l1162_116216

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 30 →
  history_A = 7 →
  math_A = 13 →
  both_A = 4 →
  total - ((history_A + math_A) - both_A) = 14 :=
by sorry

end students_without_A_l1162_116216


namespace polynomial_coefficient_sums_l1162_116276

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129) ∧
  (a₁ + a₃ + a₅ + a₇ = 8256) ∧
  (a₀ + a₂ + a₄ + a₆ = -8128) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 16384) := by
  sorry

end polynomial_coefficient_sums_l1162_116276


namespace range_of_a_when_proposition_false_l1162_116284

theorem range_of_a_when_proposition_false (a : ℝ) :
  (∀ t : ℝ, t^2 - 2*t - a ≥ 0) → a ≤ -1 := by
  sorry

end range_of_a_when_proposition_false_l1162_116284


namespace a_outside_interval_l1162_116272

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem a_outside_interval (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_nonpositive f) 
  (h_inequality : f a > f 2) : 
  a < -2 ∨ a > 2 :=
sorry

end a_outside_interval_l1162_116272


namespace intersection_distance_l1162_116248

/-- The distance between the intersection points of two curves in polar coordinates -/
theorem intersection_distance (θ : Real) : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (ρ : ℝ), ρ * Real.sin (θ + π/4) = 1 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    (∀ (ρ : ℝ), ρ = Real.sqrt 2 → (ρ * Real.cos θ, ρ * Real.sin θ) = A ∨ (ρ * Real.cos θ, ρ * Real.sin θ) = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end intersection_distance_l1162_116248


namespace intersection_of_P_and_Q_l1162_116291

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-1, 0} := by
  sorry

end intersection_of_P_and_Q_l1162_116291


namespace min_value_of_function_l1162_116297

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ ∃ t > 0, (t^2 - 4*t + 1) / t = -2 :=
sorry

end min_value_of_function_l1162_116297


namespace arithmetic_sequence_problem_l1162_116255

theorem arithmetic_sequence_problem (a b c d e : ℕ) :
  a < 10 ∧
  b = 12 ∧
  e = 33 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  e < 100 ∧
  b - a = c - b ∧
  c - b = d - c ∧
  d - c = e - d →
  a = 5 ∧ b = 12 ∧ c = 19 ∧ d = 26 ∧ e = 33 :=
by sorry

end arithmetic_sequence_problem_l1162_116255


namespace circle_line_intersection_l1162_116292

def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

theorem circle_line_intersection (r : ℝ) (hr : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ A ∈ Line ∧
    B ∈ Circle r ∧ B ∈ Line ∧
    C ∈ Circle r ∧
    C.1 = (5/4 * A.1 + 3/4 * B.1) ∧
    C.2 = (5/4 * A.2 + 3/4 * B.2) →
  r = Real.sqrt 10 := by
sorry

end circle_line_intersection_l1162_116292


namespace flea_treatment_l1162_116213

theorem flea_treatment (initial_fleas : ℕ) : 
  (initial_fleas / 2 / 2 / 2 / 2 = 14) → (initial_fleas - 14 = 210) := by
  sorry

end flea_treatment_l1162_116213


namespace smallest_special_number_l1162_116259

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n % 20 = 0 → is_perfect_cube (n^2) → is_perfect_square (n^3) → 
  n ≥ 1000000 :=
sorry

end smallest_special_number_l1162_116259


namespace ellipse_m_value_l1162_116220

/-- Represents an ellipse with equation x²/(m-2) + y²/(10-m) = 1 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (m - 2) + y^2 / (10 - m) = 1

/-- Represents the focal distance of an ellipse -/
def focalDistance (e : Ellipse m) := 4

/-- Represents that the foci of the ellipse are on the x-axis -/
def fociOnXAxis (e : Ellipse m) := True

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focalDistance e = 4) (h2 : fociOnXAxis e) : m = 8 := by
  sorry

end ellipse_m_value_l1162_116220


namespace slope_plus_intercept_equals_two_thirds_l1162_116204

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (m, b)

-- Theorem statement
theorem slope_plus_intercept_equals_two_thirds :
  let (m, b) := line_through_points 2 (-1) (-1) 4
  m + b = 2 / 3 := by sorry

end slope_plus_intercept_equals_two_thirds_l1162_116204


namespace arithmetic_sequence_sum_l1162_116243

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in an arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 := by
sorry

end arithmetic_sequence_sum_l1162_116243


namespace count_four_digit_integers_eq_sixteen_l1162_116251

/-- The number of four-digit positive integers composed only of digits 2 and 5 -/
def count_four_digit_integers : ℕ :=
  let digit_choices := 2  -- number of choices for each digit (2 or 5)
  let num_digits := 4     -- number of digits in the integer
  digit_choices ^ num_digits

/-- Theorem stating that the count of four-digit positive integers
    composed only of digits 2 and 5 is equal to 16 -/
theorem count_four_digit_integers_eq_sixteen :
  count_four_digit_integers = 16 := by
  sorry

end count_four_digit_integers_eq_sixteen_l1162_116251


namespace cos_of_tan_in_third_quadrant_l1162_116203

/-- Prove that for an angle α in the third quadrant with tan α = 4/3, cos α = -3/5 -/
theorem cos_of_tan_in_third_quadrant (α : Real) 
  (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
  (h2 : Real.tan α = 4/3) : 
  Real.cos α = -3/5 := by
  sorry

end cos_of_tan_in_third_quadrant_l1162_116203


namespace base7_difference_to_decimal_l1162_116271

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers --/
def base7Difference (a b : List Nat) : List Nat :=
  sorry -- Implementation of base 7 subtraction

theorem base7_difference_to_decimal : 
  let a := [4, 1, 2, 3] -- 3214 in base 7 (least significant digit first)
  let b := [4, 3, 2, 1] -- 1234 in base 7 (least significant digit first)
  base7ToDecimal (base7Difference a b) = 721 := by
  sorry

end base7_difference_to_decimal_l1162_116271


namespace even_function_inequality_l1162_116230

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on a set if f(x) ≤ f(y) whenever x ≤ y in that set -/
def MonoIncOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

/-- The theorem statement -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_mono : MonoIncOn f (Set.Ici 0))
    (h_ineq : ∀ x ∈ Set.Icc (1/2) 1, f (a*x + 1) - f (x - 2) ≤ 0) :
  a ∈ Set.Icc (-2) 0 := by
  sorry

end even_function_inequality_l1162_116230


namespace part_one_part_two_l1162_116237

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3*a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x > 5 - |2*x - 1|} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Part II
theorem part_two (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ + x₀ < 6) → a < 2 := by sorry

end part_one_part_two_l1162_116237


namespace even_function_theorem_l1162_116240

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_theorem (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_positive : ∀ x > 0, f x = (1 - x) * x) : 
  ∀ x < 0, f x = -x^2 - x := by
sorry

end even_function_theorem_l1162_116240


namespace fraction_equivalence_l1162_116223

theorem fraction_equivalence : 8 / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end fraction_equivalence_l1162_116223


namespace baso4_percentage_yield_is_90_percent_l1162_116254

-- Define the molar quantities
def NaOH_moles : ℚ := 3
def H2SO4_moles : ℚ := 2
def BaCl2_moles : ℚ := 1
def BaSO4_actual_yield : ℚ := 9/10

-- Define the reaction stoichiometry
def NaOH_to_Na2SO4_ratio : ℚ := 2
def H2SO4_to_Na2SO4_ratio : ℚ := 1
def Na2SO4_to_BaSO4_ratio : ℚ := 1
def BaCl2_to_BaSO4_ratio : ℚ := 1

-- Define the theoretical yield calculation
def theoretical_yield (limiting_reactant_moles ratio : ℚ) : ℚ :=
  limiting_reactant_moles / ratio

-- Define the percentage yield calculation
def percentage_yield (actual_yield theoretical_yield : ℚ) : ℚ :=
  actual_yield / theoretical_yield * 100

-- Theorem to prove
theorem baso4_percentage_yield_is_90_percent :
  let na2so4_yield_from_naoh := theoretical_yield NaOH_moles NaOH_to_Na2SO4_ratio
  let na2so4_yield_from_h2so4 := theoretical_yield H2SO4_moles H2SO4_to_Na2SO4_ratio
  let na2so4_actual_yield := min na2so4_yield_from_naoh na2so4_yield_from_h2so4
  let baso4_theoretical_yield := min na2so4_actual_yield BaCl2_moles
  percentage_yield BaSO4_actual_yield baso4_theoretical_yield = 90 :=
by sorry

end baso4_percentage_yield_is_90_percent_l1162_116254


namespace hiram_allyson_age_problem_l1162_116283

/-- The number added to Hiram's age -/
def x : ℕ := 12

theorem hiram_allyson_age_problem :
  let hiram_age : ℕ := 40
  let allyson_age : ℕ := 28
  hiram_age + x = 2 * allyson_age - 4 :=
by sorry

end hiram_allyson_age_problem_l1162_116283


namespace inequality_solution_set_l1162_116227

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-1 : ℝ) 3) = {x | (3 - x) * (1 + x) > 0} := by sorry

end inequality_solution_set_l1162_116227


namespace parallel_segment_length_l1162_116288

/-- Represents a trapezoid with given base lengths -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  h : shorter_base > 0
  k : longer_base > shorter_base

/-- Represents a line segment parallel to the bases of a trapezoid -/
structure ParallelSegment (T : Trapezoid) where
  length : ℝ
  passes_through_diagonal_intersection : Bool

/-- The theorem statement -/
theorem parallel_segment_length 
  (T : Trapezoid) 
  (S : ParallelSegment T) 
  (h : T.shorter_base = 4) 
  (k : T.longer_base = 12) 
  (m : S.passes_through_diagonal_intersection = true) : 
  S.length = 6 := by
  sorry

end parallel_segment_length_l1162_116288


namespace pyramid_base_theorem_l1162_116228

def isPyramidBase (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def pyramidTop (a b c d e : ℕ) : ℕ :=
  a * b^4 * c^6 * d^4 * e

theorem pyramid_base_theorem (a b c d e : ℕ) :
  isPyramidBase a b c d e ∧ pyramidTop a b c d e = 140026320 →
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨
   (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨
   (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨
   (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1)) :=
by sorry

end pyramid_base_theorem_l1162_116228


namespace dans_balloons_l1162_116219

theorem dans_balloons (sam_initial : Real) (fred_given : Real) (total : Real) : Real :=
  let sam_remaining := sam_initial - fred_given
  let dan_balloons := total - sam_remaining
  dan_balloons

#check dans_balloons 46.0 10.0 52.0

end dans_balloons_l1162_116219


namespace consecutive_numbers_percentage_l1162_116242

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g = 7 * 9) →
  (b = a + 1) →
  (c = b + 1) →
  (d = c + 1) →
  (e = d + 1) →
  (f = e + 1) →
  (g = f + 1) →
  (a : ℚ) / g * 100 = 50 := by
sorry

end consecutive_numbers_percentage_l1162_116242


namespace cone_base_circumference_l1162_116222

/-- Theorem: For a right circular cone with volume 24π cubic centimeters and height 6 cm, 
    the circumference of its base is 4√3π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h → 
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end cone_base_circumference_l1162_116222


namespace jessies_weight_calculation_l1162_116281

/-- Calculates Jessie's current weight after changes due to jogging, diet, and strength training -/
def jessies_current_weight (initial_weight weight_lost_jogging weight_lost_diet weight_gained_training : ℕ) : ℕ :=
  initial_weight - weight_lost_jogging - weight_lost_diet + weight_gained_training

/-- Theorem stating that Jessie's current weight is 29 kilograms -/
theorem jessies_weight_calculation :
  jessies_current_weight 69 35 10 5 = 29 := by
  sorry

end jessies_weight_calculation_l1162_116281


namespace least_n_satisfying_inequality_l1162_116226

theorem least_n_satisfying_inequality : 
  (∃ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ+, (1 : ℚ) / m - (1 : ℚ) / (m + 2) < (1 : ℚ) / 15 → m ≥ 6) :=
by sorry

end least_n_satisfying_inequality_l1162_116226


namespace complex_equation_solution_l1162_116224

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) →
  z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) :=
by sorry

end complex_equation_solution_l1162_116224


namespace rectangle_dimension_change_l1162_116277

/-- Theorem: If the length of a rectangle is increased by 50% and the area remains constant, 
    then the width of the rectangle must be decreased by 33.33%. -/
theorem rectangle_dimension_change (L W A : ℝ) (h1 : A = L * W) (h2 : A > 0) (h3 : L > 0) (h4 : W > 0) :
  let new_L := 1.5 * L
  let new_W := A / new_L
  (W - new_W) / W = 1 / 3 := by
  sorry

end rectangle_dimension_change_l1162_116277


namespace jungkook_has_fewest_erasers_l1162_116280

/-- The number of erasers Jungkook has -/
def jungkook_erasers : ℕ := 6

/-- The number of erasers Jimin has -/
def jimin_erasers : ℕ := jungkook_erasers + 4

/-- The number of erasers Seokjin has -/
def seokjin_erasers : ℕ := jimin_erasers - 3

theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end jungkook_has_fewest_erasers_l1162_116280


namespace increase_by_percentage_increase_75_by_150_percent_l1162_116260

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) := by sorry

theorem increase_75_by_150_percent :
  75 * (1 + 150 / 100) = 187.5 := by sorry

end increase_by_percentage_increase_75_by_150_percent_l1162_116260


namespace unique_solution_sqrt_two_equation_l1162_116268

theorem unique_solution_sqrt_two_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 := by
  sorry

end unique_solution_sqrt_two_equation_l1162_116268


namespace expand_product_l1162_116256

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l1162_116256


namespace total_candle_weight_l1162_116205

/-- Represents the composition of a candle in ounces -/
structure CandleComposition where
  beeswax : ℝ
  coconut_oil : ℝ
  essential_oils : ℝ

/-- Calculates the total weight of a candle given its composition -/
def candle_weight (c : CandleComposition) : ℝ :=
  c.beeswax + c.coconut_oil + c.essential_oils

/-- Defines the composition of a small candle -/
def small_candle : CandleComposition :=
  { beeswax := 4, coconut_oil := 2, essential_oils := 0.5 }

/-- Defines the composition of a medium candle -/
def medium_candle : CandleComposition :=
  { beeswax := 8, coconut_oil := 1, essential_oils := 1 }

/-- Defines the composition of a large candle -/
def large_candle : CandleComposition :=
  { beeswax := 16, coconut_oil := 3, essential_oils := 2 }

/-- The number of small candles made -/
def num_small_candles : ℕ := 4

/-- The number of medium candles made -/
def num_medium_candles : ℕ := 3

/-- The number of large candles made -/
def num_large_candles : ℕ := 2

/-- Theorem stating that the total weight of all candles is 98 ounces -/
theorem total_candle_weight :
  (num_small_candles : ℝ) * candle_weight small_candle +
  (num_medium_candles : ℝ) * candle_weight medium_candle +
  (num_large_candles : ℝ) * candle_weight large_candle = 98 := by
  sorry

end total_candle_weight_l1162_116205


namespace tim_has_twelve_nickels_l1162_116239

/-- Represents the number of coins Tim has -/
structure TimsCoins where
  quarters : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total number of nickels Tim has after receiving coins from his dad -/
def total_nickels (initial : TimsCoins) (from_dad : TimsCoins) : ℕ :=
  initial.nickels + from_dad.nickels

/-- Theorem stating that Tim has 12 nickels after receiving coins from his dad -/
theorem tim_has_twelve_nickels :
  let initial := TimsCoins.mk 7 9 0
  let from_dad := TimsCoins.mk 0 3 5
  total_nickels initial from_dad = 12 := by
  sorry


end tim_has_twelve_nickels_l1162_116239


namespace tourist_arrangement_count_l1162_116208

/-- The number of tourists --/
def num_tourists : ℕ := 5

/-- The number of scenic spots --/
def num_spots : ℕ := 4

/-- The function to calculate the number of valid arrangements --/
def valid_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  k^n - Nat.choose k 1 * (k-1)^n + Nat.choose k 2 * (k-2)^n - Nat.choose k 3 * (k-3)^n

/-- The main theorem to prove --/
theorem tourist_arrangement_count :
  (valid_arrangements num_tourists num_spots) * (num_spots - 1) * (num_spots - 1) / num_spots = 216 :=
sorry

end tourist_arrangement_count_l1162_116208
